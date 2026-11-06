"""Locked, no-optimization evaluation for fusion.research_pattern.

CSV columns: timestamp,open,high,low,close,spread_pips. Uses chronological
first/second halves as separate evidence cells; no tuning is performed. Each
trade enters at the following bar open, exits at first barrier with stop-first
tie handling, or at a 120-minute timeout. Costs are charged once per trade,
matching the project's simplified spread model. This is still a candle-level
approximation, not broker execution.
"""
from __future__ import annotations
import argparse, csv, math, sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion.research_pattern import (
    EMA_FAST, EMA_SLOW, EMA_SLOPE_LOOKBACK, ATR_PERIOD, RSI_PERIOD, PIP,
    STOP_ATR, TARGET_ATR, MIN_ATR_TO_SPREAD, MAX_HOLD_MINUTES,
)

@dataclass
class Bar:
    timestamp: str
    o: float; h: float; l: float; c: float; spread: float


def ema(values, period):
    alpha=2/(period+1); out=[values[0]]
    for x in values[1:]: out.append(alpha*x+(1-alpha)*out[-1])
    return out


def features(bars):
    close=[b.c for b in bars]
    e20,e50=ema(close,EMA_FAST),ema(close,EMA_SLOW)
    tr=[]
    for i,b in enumerate(bars):
        pc=close[i-1] if i else b.c
        tr.append(max(b.h-b.l,abs(b.h-pc),abs(b.l-pc)))
    atr=[None]*len(bars)
    for i in range(ATR_PERIOD-1,len(bars)):
        atr[i]=sum(tr[i-ATR_PERIOD+1:i+1])/ATR_PERIOD
    rsi=[None]*len(bars)
    for i in range(RSI_PERIOD,len(bars)):
        changes=[close[j]-close[j-1] for j in range(i-RSI_PERIOD+1,i+1)]
        gain=sum(max(x,0) for x in changes)/RSI_PERIOD
        loss=sum(max(-x,0) for x in changes)/RSI_PERIOD
        rsi[i]=100 if loss==0 and gain else (50 if loss==0 else 100-100/(1+gain/loss))
    return e20,e50,atr,rsi


def wilson(k,n):
    if not n:return (0.0,0.0)
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*(c-h),100*(c+h)


def run(bars, start, end, fixed_cost=None):
    e20,e50,atr,rsi=features(bars)
    trades=[]; i=max(start,EMA_SLOW+EMA_SLOPE_LOOKBACK)
    while i+1<end:
        if atr[i] is None or rsi[i] is None:
            i+=1;continue
        cost=bars[i].spread if fixed_cost is None else fixed_cost
        if cost<0 or cost>1.2 or atr[i]/PIP < MIN_ATR_TO_SPREAD*cost:
            i+=1;continue
        b=bars[i]; fast=e20[i]
        long=(e20[i]>e50[i] and fast>e20[i-EMA_SLOPE_LOOKBACK]
              and b.l<=fast<=b.c and b.c>b.o and 45<=rsi[i]<=65)
        short=(e20[i]<e50[i] and fast<e20[i-EMA_SLOPE_LOOKBACK]
               and b.h>=fast>=b.c and b.c<b.o and 35<=rsi[i]<=55)
        if long==short:
            i+=1;continue
        side=1 if long else -1
        entry=bars[i+1].o
        stop=STOP_ATR*atr[i];target=TARGET_ATR*atr[i]
        finish=min(i+1+MAX_HOLD_MINUTES,end)
        gross=None;exit_i=None
        for j in range(i+1,finish):
            q=bars[j]
            hs=q.l<=entry-stop if side>0 else q.h>=entry+stop
            ht=q.h>=entry+target if side>0 else q.l<=entry-target
            if hs:
                gross=-stop/PIP;exit_i=j;break
            if ht:
                gross=target/PIP;exit_i=j;break
        if gross is None:
            exit_i=finish-1
            gross=side*(bars[exit_i].c-entry)/PIP
        net=gross-cost
        trades.append((net,gross,cost,exit_i-(i+1)+1,
                       100*(stop/PIP+cost)/(stop/PIP+target/PIP)))
        i=exit_i+1 # one position, no same-candle re-entry
    return trades


def report(name,trades):
    n=len(trades); wins=sum(t[0]>0 for t in trades); wr=100*wins/n if n else 0
    mean=sum(t[0] for t in trades)/n if n else 0
    costs=sum(t[2] for t in trades)/n if n else 0
    # Mean per-trade binary barrier break-even (timeout exits mean this is a
    # reference, not a complete expectancy estimator). Net P&L is primary.
    be=sum(t[4] for t in trades)/n if n else 0.0
    lo,hi=wilson(wins,n);eq=peak=dd=0;streak=maxstreak=0
    for net,_,_,_,_ in trades:
        eq+=net;peak=max(peak,eq);dd=max(dd,peak-eq)
        streak=streak+1 if net<=0 else 0;maxstreak=max(maxstreak,streak)
    print(f"{name}: n={n} wins={wr:.2f}% Wilson95=[{lo:.2f},{hi:.2f}] "
          f"barrier_BE_mean={be:.2f}% mean_net={mean:+.3f}p mean_cost={costs:.3f}p "
          f"worst={min((t[0] for t in trades),default=0):+.2f}p maxDD={dd:.1f}p "
          f"max_loss_run={maxstreak} avg_hold={sum(t[3] for t in trades)/n if n else 0:.1f}m")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',required=True,type=Path)
    p.add_argument('--fixed-cost',type=float,default=None,
                   help='stress cost in pips; default uses observed signal-bar spread')
    a=p.parse_args()
    bars=[]
    with a.csv.open(newline='') as f:
        for r in csv.DictReader(f):
            bars.append(Bar(r['timestamp'],float(r['open']),float(r['high']),
                            float(r['low']),float(r['close']),float(r['spread_pips'])))
    cut=len(bars)//2
    print(f"bars={len(bars)} period={bars[0].timestamp}..{bars[-1].timestamp} "
          f"mean_spread={sum(b.spread for b in bars)/len(bars):.3f}p cost_mode="
          f"{'observed' if a.fixed_cost is None else str(a.fixed_cost)+'p stress'}")
    print(f"split: first={bars[0].timestamp}..{bars[cut-1].timestamp}; "
          f"held-out={bars[cut].timestamp}..{bars[-1].timestamp}")
    report('first half',run(bars,0,cut,a.fixed_cost))
    report('second half held-out',run(bars,cut,len(bars),a.fixed_cost))

if __name__=='__main__': main()
