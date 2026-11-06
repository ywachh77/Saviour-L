"""Independent verification of the scalp_expansion claim on tick-derived bars.

This does not tune the rule. It runs the checked-in scalp_expansion function at
1/2/3/5-minute close exits, with next-bar-open entry, on each chronological half.
Input CSV: timestamp,open,high,low,close,spread_pips. Spread is charged once per
trade; an observed-mean and 1.2-pip cost scenario are both reported.
"""
from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fusion.domain import Candle
from fusion import scalping
from fusion.indicators import atr as atr_series, ema, mean, pstdev
from fusion.instruments import for_symbol

HORIZONS=(1,2,3,5)
PIP=0.0001


def wilson(k,n):
    if n==0:return (0.0,0.0)
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*(c-h),100*(c+h)


def minimal_features(candles):
    """Compute only scalp_expansion inputs, matching production definitions."""
    closes=[c.close for c in candles]
    e_fast=ema(closes,5);e_slow=ema(closes,20);atr=atr_series(candles,14)
    result=[]
    for i,c in enumerate(candles):
        if i<30 or e_fast[i] is None or e_slow[i] is None or atr[i] is None:
            result.append(None);continue
        window=closes[i-19:i+1]
        band_mid=mean(window);band_width=2*pstdev(window)
        prior=candles[i-20:i]
        r=c.high-c.low
        result.append({
            "atr_pips":atr[i]/PIP,
            "close_position":(c.close-c.low)/r if r>0 else 0.5,
            "ema_fast":e_fast[i],"ema_slow":e_slow[i],
            "bb_upper":band_mid+band_width,"bb_lower":band_mid-band_width,
            "prior_high_20":max(x.high for x in prior),
            "prior_low_20":min(x.low for x in prior),
        })
    return result


def evaluate(candles,features,cost,horizon):
    nets=[];grosses=[];i=1;last=len(candles)-1
    while i+horizon<=last:
        f=features[i]
        if f is None:
            i+=1;continue
        decision=scalping.scalp_expansion(candles[i],candles[i-1].close,f,None,spread_pips=cost)
        if decision.side not in ("BUY","SELL"):
            i+=1;continue
        entry=candles[i+1].open;exit_price=candles[i+horizon].close
        gross=(exit_price-entry)/PIP*(1 if decision.side=="BUY" else -1)
        grosses.append(gross);nets.append(gross-cost)
        # One trade at a time; candidate signals while holding are ignored.
        i+=horizon+1
    n=len(nets);wins=sum(x>0 for x in nets);lo,hi=wilson(wins,n)
    mean_net=sum(nets)/n if n else 0.0
    gw=[x for x in grosses if x>0];gl=[-x for x in grosses if x<0]
    avgw=sum(gw)/len(gw) if gw else 0.0;avgl=sum(gl)/len(gl) if gl else 0.0
    be=100*(avgl+cost)/(avgw+avgl) if avgw+avgl else 0.0
    equity=peak=dd=0.0;run=longest=0
    for x in nets:
        equity+=x;peak=max(peak,equity);dd=max(dd,peak-equity)
        run=run+1 if x<=0 else 0;longest=max(longest,run)
    return (n,100*wins/n if n else 0,lo,hi,be,mean_net,
            min(nets,default=0),dd,longest)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',required=True,type=Path)
    a=p.parse_args();rows=[]
    with a.csv.open(newline='') as f:
        for r in csv.DictReader(f):
            rows.append((r['timestamp'],float(r['open']),float(r['high']),float(r['low']),float(r['close']),float(r['spread_pips'])))
    from datetime import datetime
    candles=[Candle(datetime.fromisoformat(t),o,h,l,c) for t,o,h,l,c,_ in rows]
    spreads=[x[5] for x in rows]
    print(f"bars={len(candles)} period={rows[0][0]}..{rows[-1][0]} mean_spread={sum(spreads)/len(spreads):.3f}p")
    cut=len(candles)//2
    all_features=minimal_features(candles)
    for lo,hi,label in ((0,cut,'first half'),(cut,len(candles),'second half (held out)')):
        part=candles[lo:hi]
        fts=all_features[lo:hi]
        costs=(sum(spreads[lo:hi])/(hi-lo),1.2)
        print(label, f"{part[0].timestamp.isoformat()}..{part[-1].timestamp.isoformat()} mean_spread={costs[0]:.3f}p")
        for cost in costs:
            print(f"  cost={cost:.3f}p")
            for h in HORIZONS:
                n,wr,wlo,whi,be,net,worst,dd,streak=evaluate(part,fts,cost,h)
                print(f"    h={h}m n={n} net_win={wr:.2f}% Wilson=[{wlo:.2f},{whi:.2f}] "
                      f"BE={be:.2f}% net/trade={net:+.3f}p worst={worst:+.2f}p "
                      f"maxDD={dd:.1f}p loss_run={streak}")

if __name__=='__main__': main()
