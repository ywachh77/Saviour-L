"""Paper-test the idea of seven consecutive net-positive trades in one hour.

CSV columns: timestamp,open,high,low,close,spread_pips. Compares the isolated
trend-pullback candidate and the checked-in scalp_expansion prototype. This is
historical simulation only; never routes broker orders. Costs are charged once
per trade, midpoint OHLC is used, and ambiguous barrier bars resolve stop-first.
"""
from __future__ import annotations
import argparse,csv,math,sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from fusion.domain import Candle
from fusion.indicators import atr as atr_series,ema,mean,pstdev
from fusion import scalping
from fusion.research_pattern import (EMA_FAST,EMA_SLOW,EMA_SLOPE_LOOKBACK,
    ATR_PERIOD,RSI_PERIOD,STOP_ATR,TARGET_ATR,MAX_HOLD_MINUTES,MIN_ATR_TO_SPREAD,PIP)

@dataclass
class Trade:
    entry:datetime
    exit:datetime
    gross:float
    cost:float
    @property
    def net(self):return self.gross-self.cost


def wilson(k,n):
    if not n:return (0.0,0.0)
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*(c-h),100*(c+h)


def features_for_scalp(cs):
    closes=[c.close for c in cs];fast=ema(closes,5);slow=ema(closes,20);ats=atr_series(cs,14);out=[]
    for i,c in enumerate(cs):
        if i<30 or fast[i] is None or slow[i] is None or ats[i] is None:out.append(None);continue
        win=closes[i-19:i+1];prior=cs[i-20:i];rng=c.high-c.low
        out.append({'atr_pips':ats[i]/PIP,'close_position':(c.close-c.low)/rng if rng else .5,
            'ema_fast':fast[i],'ema_slow':slow[i],'bb_upper':mean(win)+2*pstdev(win),
            'bb_lower':mean(win)-2*pstdev(win),'prior_high_20':max(x.high for x in prior),
            'prior_low_20':min(x.low for x in prior)})
    return out


def scalp_trades(cs,fts,cost,h=3):
    out=[];i=30
    while i+h<len(cs):
        f=fts[i]
        if f is None:i+=1;continue
        d=scalping.scalp_expansion(cs[i],cs[i-1].close,f,None,spread_pips=cost)
        if d.side not in ('BUY','SELL'):i+=1;continue
        entry=cs[i+1].open;exitc=cs[i+h].close
        gross=(exitc-entry)/PIP*(1 if d.side=='BUY' else -1)
        out.append(Trade(cs[i+1].timestamp,cs[i+h].timestamp+timedelta(minutes=1),gross,cost))
        i+=h+1
    return out


def trend_trades(cs,cost):
    closes=[c.close for c in cs];fast=ema(closes,EMA_FAST);slow=ema(closes,EMA_SLOW)
    ats=atr_series(cs,ATR_PERIOD);rsis=[None]*len(cs)
    for i in range(RSI_PERIOD,len(cs)):
        changes=[closes[j]-closes[j-1] for j in range(i-RSI_PERIOD+1,i+1)]
        g=sum(max(0,x) for x in changes)/RSI_PERIOD;l=sum(max(0,-x) for x in changes)/RSI_PERIOD
        rsis[i]=100 if l==0 and g else (50 if l==0 else 100-100/(1+g/l))
    out=[];i=EMA_SLOW+EMA_SLOPE_LOOKBACK
    while i+1<len(cs):
        b=cs[i];a=ats[i];r=rsis[i]
        if a is None or r is None or cost>1.2 or a/PIP<MIN_ATR_TO_SPREAD*cost:
            i+=1;continue
        up=fast[i]>slow[i] and fast[i]>fast[i-EMA_SLOPE_LOOKBACK] and b.low<=fast[i]<=b.close and b.close>b.open and 45<=r<=65
        dn=fast[i]<slow[i] and fast[i]<fast[i-EMA_SLOPE_LOOKBACK] and b.high>=fast[i]>=b.close and b.close<b.open and 35<=r<=55
        if up==dn:i+=1;continue
        side=1 if up else -1;entry=cs[i+1].open;st=STOP_ATR*a;tp=TARGET_ATR*a
        end=min(i+1+MAX_HOLD_MINUTES,len(cs));gross=None;ei=None
        for j in range(i+1,end):
            q=cs[j];hs=q.low<=entry-st if side>0 else q.high>=entry+st
            ht=q.high>=entry+tp if side>0 else q.low<=entry-tp
            if hs:gross=-st/PIP;ei=j;break
            if ht:gross=tp/PIP;ei=j;break
        if gross is None:ei=end-1;gross=side*(cs[ei].close-entry)/PIP
        out.append(Trade(cs[i+1].timestamp,cs[ei].timestamp+timedelta(minutes=1),gross,cost))
        i=ei+1
    return out


def summarize(name,trades,active_hours):
    total_hours=max(1,len(active_hours))
    by_hour=defaultdict(list)
    eligible=[]
    for t in trades:
        # Assign every executed trade to its entry hour. To qualify, at least
        # seven trades must start and settle within the same observed UTC hour,
        # and every trade entered that hour must be net-positive.
        bucket=t.entry.replace(minute=0,second=0,microsecond=0)
        by_hour[bucket].append(t)
        eligible.append(t)
    qualifies=sum(len(v)>=7 and all(t.net>0 and t.exit < h+timedelta(hours=1) for t in v)
                  for h,v in by_hour.items())
    # Count non-overlapping seven-consecutive-win streaks whose first-to-last
    # trade interval is <=60m. Greedy non-overlap prevents rolling-window inflation.
    seq=0;i=0
    while i+6<len(eligible):
        block=eligible[i:i+7]
        if all(t.net>0 for t in block) and (block[-1].exit-block[0].entry)<=timedelta(hours=1):
            seq+=1;i+=7
        else:i+=1
    n=len(trades);wins=sum(t.net>0 for t in trades);lo,hi=wilson(wins,n)
    expectancy=sum(t.net for t in trades)/n if n else 0
    gross_wins=[t.gross for t in trades if t.gross>0]
    gross_losses=[-t.gross for t in trades if t.gross<0]
    avg_win=sum(gross_wins)/len(gross_wins) if gross_wins else 0
    avg_loss=sum(gross_losses)/len(gross_losses) if gross_losses else 0
    avg_cost=sum(t.cost for t in trades)/n if n else 0
    be=100*(avg_loss+avg_cost)/(avg_win+avg_loss) if avg_win+avg_loss else 0
    max_run=run=0;eq=peak=dd=0
    for t in trades:
        run=run+1 if t.net<=0 else 0;max_run=max(max_run,run)
        eq+=t.net;peak=max(peak,eq);dd=max(dd,peak-eq)
    print(f'{name}: trades={n}; net win={100*wins/n if n else 0:.2f}% '
          f'Wilson=[{lo:.2f},{hi:.2f}] break_even≈{be:.2f}% mean_net={expectancy:+.3f}p; '
          f'max loss run={max_run}; maxDD={dd:.1f}p')
    eligible_hours=sum(len(v)>=7 and all(t.exit < h+timedelta(hours=1) for t in v)
                        for h,v in by_hour.items())
    print(f'  observed UTC hours={total_hours}; hours with >=7 trades, all settled within hour, '
          f'and zero losses={qualifies} ({100*qualifies/total_hours:.4f}%); '
          f'hours with >=7 entries settled within hour={eligible_hours}; '
          f'non-overlap 7-win sequences <=60m={seq}')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--csv',required=True,type=Path);a=ap.parse_args()
    rows=list(csv.DictReader(a.csv.open(newline='')))
    cs=[Candle(datetime.fromisoformat(r['timestamp']),float(r['open']),float(r['high']),float(r['low']),float(r['close'])) for r in rows]
    spreads=[float(r['spread_pips']) for r in rows];cut=len(cs)//2
    first_hour=cs[0].timestamp.replace(minute=0,second=0,microsecond=0)
    last_hour=cs[-1].timestamp.replace(minute=0,second=0,microsecond=0)
    fts=features_for_scalp(cs)
    print(f'bars={len(cs)} {cs[0].timestamp}..{cs[-1].timestamp}; UTC hour blocks; '
          f'mean tick spread={sum(spreads)/len(spreads):.3f}p')
    for lo,hi,label in ((0,cut,'first half'),(cut,len(cs),'second half')):
        part=cs[lo:hi];partfts=fts[lo:hi];mcost=sum(spreads[lo:hi])/(hi-lo)
        print('\n'+label,part[0].timestamp,part[-1].timestamp)
        active_hours={c.timestamp.replace(minute=0,second=0,microsecond=0) for c in part}
        for cost in (mcost,1.2):
            print(f'  cost={cost:.3f}p')
            summarize('trend-pullback',trend_trades(part,cost),active_hours)
            summarize('scalp-expansion 3m',scalp_trades(part,partfts,cost),active_hours)
if __name__=='__main__':main()
