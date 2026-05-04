import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from collections import deque
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Jarvis Pro — Crypto", page_icon="🧠", layout="wide")

BIN = "https://api.binance.com/api/v3"
KUC = "https://api.kucoin.com/api/v1"
OKX = "https://www.okx.com/api/v5"
HDR = {"User-Agent": "Mozilla/5.0"}

TF_BIN = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w"]
TF_KUC = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min",
           "1h":"1hour","2h":"2hour","4h":"4hour","6h":"6hour","8h":"8hour",
           "12h":"12hour","1d":"1day","1w":"1week"}
TF_OKX = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
           "1h":"1H","2h":"2H","4h":"4H","6h":"6H","8h":"8H",
           "12h":"12H","1d":"1D","1w":"1W"}
TF_HIGHER = {"1m":"15m","3m":"30m","5m":"1h","15m":"4h","30m":"4h",
             "1h":"4h","2h":"4h","4h":"1d","6h":"1d","8h":"1d",
             "12h":"1d","1d":"1w","3d":"1w","1w":"1w"}
EX_ICONS = {"Binance":"🔵","KuCoin":"🟠","OKX":"🟣"}
EX_CLR   = {"Binance":"#00b4d8","KuCoin":"#f77f00","OKX":"#a855f7"}

# ── COIN LIST ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300,show_spinner=False)
def get_all_coins():
    seen={}
    def _bin():
        try:
            r=requests.get(f"{BIN}/ticker/24hr",timeout=12)
            if r.status_code!=200: return []
            return [{"exchange":"Binance","symbol":x["symbol"][:-4],
                     "pair":x["symbol"],"price":float(x.get("lastPrice",0) or 0),
                     "chg":float(x.get("priceChangePercent",0) or 0),
                     "vol":float(x.get("quoteVolume",0) or 0)}
                    for x in r.json() if isinstance(x,dict) and x.get("symbol","").endswith("USDT")]
        except: return []
    def _kuc():
        try:
            r=requests.get(f"{KUC}/market/allTickers",timeout=12)
            if r.status_code!=200: return []
            return [{"exchange":"KuCoin","symbol":x["symbol"][:-5],
                     "pair":x["symbol"],"price":float(x.get("last",0) or 0),
                     "chg":float(x.get("changeRate",0) or 0)*100,
                     "vol":float(x.get("volValue",0) or 0)}
                    for x in r.json().get("data",{}).get("ticker",[])
                    if isinstance(x,dict) and x.get("symbol","").endswith("-USDT")]
        except: return []
    def _okx():
        try:
            r=requests.get(f"{OKX}/market/tickers",params={"instType":"SPOT"},timeout=12)
            if r.status_code!=200: return []
            out=[]
            for x in r.json().get("data",[]):
                if not isinstance(x,dict) or not x.get("instId","").endswith("-USDT"): continue
                px=float(x.get("last",0) or 0); op=float(x.get("open24h",px) or px)
                out.append({"exchange":"OKX","symbol":x["instId"][:-5],"pair":x["instId"],
                            "price":px,"chg":((px/op)-1)*100 if op else 0,
                            "vol":float(x.get("volCcy24h",0) or 0)})
            return out
        except: return []
    for c in _bin()+_kuc()+_okx():
        s=c["symbol"]
        if s not in seen or c["vol"]>seen[s]["vol"]: seen[s]=c
    result=[]
    for c in sorted(seen.values(),key=lambda x:-x["vol"]):
        px=c["price"]
        fmt=f"${px:,.6f}" if px<0.01 else f"${px:,.4f}" if px<1 else f"${px:,.2f}"
        arr="🟢" if c["chg"]>=0 else "🔴"
        c["display"]=f"{EX_ICONS.get(c['exchange'],'⚪')} {arr} {c['symbol']} — {fmt} ({c['chg']:+.2f}%) [{c['exchange']}]"
        result.append(c)
    return result

# ── OHLCV ──────────────────────────────────────────────────────────────────

def _tfsec(tf):
    return {"1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,"1h":3600,
            "2h":7200,"4h":14400,"6h":21600,"8h":28800,"12h":43200,
            "1d":86400,"3d":259200,"1w":604800}.get(tf,3600)

@st.cache_data(ttl=60,show_spinner=False)
def _bin_ohlcv(pair,tf,n):
    try:
        r=requests.get(f"{BIN}/klines",timeout=10,params={"symbol":pair,"interval":tf,"limit":n})
        if r.status_code!=200: return pd.DataFrame()
        df=pd.DataFrame(r.json(),columns=["ts","open","high","low","close","volume","a","b","c","d","e","f"])
        for c in ["open","high","low","close","volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
        df["ts"]=pd.to_datetime(df["ts"],unit="ms"); df.set_index("ts",inplace=True)
        return df.dropna(subset=["open","high","low","close"])
    except: return pd.DataFrame()

@st.cache_data(ttl=60,show_spinner=False)
def _kuc_ohlcv(pair,tf,n):
    try:
        kf=TF_KUC.get(tf); et=int(time.time())
        r=requests.get(f"{KUC}/market/candles",timeout=10,
            params={"symbol":pair,"type":kf,"endAt":et,"startAt":et-n*_tfsec(tf)})
        if r.status_code!=200: return pd.DataFrame()
        data=r.json().get("data",[])
        if not data: return pd.DataFrame()
        df=pd.DataFrame(data,columns=["ts","open","close","high","low","volume","amount"])
        for c in ["open","high","low","close","volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
        df["ts"]=pd.to_datetime(df["ts"].astype(int),unit="s"); df.set_index("ts",inplace=True)
        return df.sort_index().dropna(subset=["open","high","low","close"])
    except: return pd.DataFrame()

@st.cache_data(ttl=60,show_spinner=False)
def _okx_ohlcv(pair,tf,n):
    try:
        okf=TF_OKX.get(tf)
        r=requests.get(f"{OKX}/market/history-candles",timeout=10,
            params={"instId":pair,"bar":okf,"limit":min(n,300)})
        if r.status_code!=200: return pd.DataFrame()
        data=r.json().get("data",[])
        if not data: return pd.DataFrame()
        df=pd.DataFrame(data,columns=["ts","open","high","low","close","vol","a","b","c"])
        for c in ["open","high","low","close","vol"]: df[c]=pd.to_numeric(df[c],errors="coerce")
        df.rename(columns={"vol":"volume"},inplace=True)
        df["ts"]=pd.to_datetime(df["ts"].astype(int),unit="ms"); df.set_index("ts",inplace=True)
        return df.sort_index().dropna(subset=["open","high","low","close"])
    except: return pd.DataFrame()

def get_ohlcv(coin,tf,n):
    ex=coin["exchange"]; sym=coin["symbol"]; pair=coin["pair"]
    order=[
        (ex, {"Binance":(_bin_ohlcv,pair),"KuCoin":(_kuc_ohlcv,pair),"OKX":(_okx_ohlcv,pair)}.get(ex,(_bin_ohlcv,f"{sym}USDT"))),
        ("Binance",(_bin_ohlcv,f"{sym}USDT")),
        ("KuCoin", (_kuc_ohlcv,f"{sym}-USDT")),
        ("OKX",    (_okx_ohlcv,f"{sym}-USDT")),
    ]
    seen_ex=set()
    for src,(fn,p) in order:
        if src in seen_ex: continue
        seen_ex.add(src)
        df=fn(p,tf,n)
        if not df.empty: return df,src
    return pd.DataFrame(),"—"

# ── INDICATORI ─────────────────────────────────────────────────────────────

def ema(s,p): return s.ewm(span=p,adjust=False).mean()
def rsi14(s,p=14):
    d=s.diff(); g=d.clip(lower=0).rolling(p).mean()
    l=(-d.clip(upper=0)).rolling(p).mean()
    return (100-(100/(1+g/l.replace(0,np.nan)))).fillna(50)
def macd_ind(s):
    m=ema(s,12)-ema(s,26); sig=m.ewm(span=9,adjust=False).mean(); return m,sig,m-sig
def boll(s,p=20,k=2):
    m=s.rolling(p).mean(); sd=s.rolling(p).std(); return m+sd*k,m-sd*k,m
def atr14(df,p=14):
    h,l,c=df["high"],df["low"],df["close"]
    return pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1).rolling(p).mean()
def adx_full(df,p=14):
    h,l,c=df["high"].values,df["low"].values,df["close"].values
    n=len(df); pdm=np.zeros(n); mdm=np.zeros(n); tra=np.zeros(n)
    for i in range(1,n):
        u=h[i]-h[i-1]; d_=l[i-1]-l[i]
        if u>d_ and u>0: pdm[i]=u
        if d_>u and d_>0: mdm[i]=d_
        tra[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    def rma(a,per):
        o=np.zeros(n); al=1/per
        if per<n: o[per]=np.mean(a[:per+1])
        for i in range(per+1,n): o[i]=o[i-1]*(1-al)+a[i]*al
        return o
    at=rma(tra,p); pdi=np.where(at>0,100*rma(pdm,p)/at,0); mdi=np.where(at>0,100*rma(mdm,p)/at,0)
    sm=pdi+mdi; dx=np.where(sm>0,100*np.abs(pdi-mdi)/sm,0); idx=df.index
    return pd.Series(rma(dx,p),index=idx),pd.Series(pdi,index=idx),pd.Series(mdi,index=idx)
def stoch14(df,k=14,d=3):
    lo=df["low"].rolling(k).min(); hi=df["high"].rolling(k).max()
    K=100*(df["close"]-lo)/(hi-lo).replace(0,np.nan)
    return K.fillna(50),K.rolling(d).mean().fillna(50)
def vwap(df):
    tp=(df["high"]+df["low"]+df["close"])/3
    return (tp*df["volume"].replace(0,np.nan)).cumsum()/df["volume"].replace(0,np.nan).cumsum()
def supertrend(df,p=10,m=3.0):
    at=atr14(df,p).values; hl=((df["high"]+df["low"])/2).values; cl=df["close"].values
    n=len(df); ub=hl+m*at; lb=hl-m*at; fub,flb=ub.copy(),lb.copy()
    d=np.ones(n,dtype=int); sv=np.full(n,np.nan)
    for i in range(1,n):
        fub[i]=min(ub[i],fub[i-1]) if cl[i-1]<=fub[i-1] else ub[i]
        flb[i]=max(lb[i],flb[i-1]) if cl[i-1]>=flb[i-1] else lb[i]
        if cl[i]>fub[i-1]: d[i]=1
        elif cl[i]<flb[i-1]: d[i]=-1
        else: d[i]=d[i-1]
        sv[i]=flb[i] if d[i]==1 else fub[i]
    return pd.Series(sv,index=df.index),pd.Series(d.astype(float),index=df.index)
def cci20(df,p=20):
    tp=(df["high"]+df["low"]+df["close"])/3; m=tp.rolling(p).mean()
    md=tp.rolling(p).apply(lambda x:np.abs(x-x.mean()).mean(),raw=True)
    return ((tp-m)/(0.015*md.replace(0,np.nan))).fillna(0)
def pivot_sr(df,n=5):
    hi=df["high"].rolling(n*2+1,center=True).max()
    lo=df["low"].rolling(n*2+1,center=True).min()
    r=df["high"][df["high"]==hi].dropna().tail(4).values.tolist()
    s=df["low"][df["low"]==lo].dropna().tail(4).values.tolist()
    return r,s

def add_all(df,eps):
    df=df.copy(); cl=df["close"]
    for p in eps: df[f"EMA_{p}"]=ema(cl,p)
    df["RSI"]=rsi14(cl)
    df["MACD"],df["MACD_sig"],df["MACD_hist"]=macd_ind(cl)
    df["BB_up"],df["BB_dn"],df["BB_mid"]=boll(cl)
    df["ATR"]=atr14(df)
    df["ADX"],df["DI_p"],df["DI_m"]=adx_full(df)
    df["SK"],df["SD"]=stoch14(df)
    df["VWAP"]=vwap(df)
    df["ST"],df["ST_d"]=supertrend(df)
    df["CCI"]=cci20(df)
    df["VOLMA"]=df["volume"].rolling(20).mean()
    return df

# ── HTF SCORE ──────────────────────────────────────────────────────────────

def _score_df(df):
    if df.empty or len(df)<30: return 50
    cl=df["close"]; sc=50
    e20=ema(cl,20).iloc[-1]; e50=ema(cl,50).iloc[-1]; pr=cl.iloc[-1]
    if pr>e20: sc+=10
    if pr>e50: sc+=10
    if e20>e50: sc+=8
    rv=rsi14(cl).iloc[-1]
    if rv>55: sc+=10
    elif rv<45: sc-=10
    m,sig,_=macd_ind(cl)
    if m.iloc[-1]>sig.iloc[-1]: sc+=10
    else: sc-=10
    _,std=supertrend(df)
    if std.iloc[-1]==1: sc+=12
    else: sc-=12
    return max(0,min(100,sc))

@st.cache_data(ttl=60,show_spinner=False)
def get_htf(coin,tf):
    htf=TF_HIGHER.get(tf,tf)
    if htf==tf: return 50,"stesso TF"
    df,_=get_ohlcv(coin,htf,80)
    return _score_df(df),htf

# ── SEGNALI FRECCE ─────────────────────────────────────────────────────────

def detect_signals(df,eps):
    if len(df)<30: return []
    cl=df["close"]; rv=rsi14(cl); m,sig,_=macd_ind(cl)
    _,std=supertrend(df); adx,pdi,mdi=adx_full(df)
    sigs=[]
    for i in range(2,len(df)):
        sc=0
        if rv.iloc[i]>30 and rv.iloc[i-1]<=30: sc+=2
        if rv.iloc[i]<70 and rv.iloc[i-1]>=70: sc-=2
        if m.iloc[i]>sig.iloc[i] and m.iloc[i-1]<=sig.iloc[i-1]: sc+=2
        if m.iloc[i]<sig.iloc[i] and m.iloc[i-1]>=sig.iloc[i-1]: sc-=2
        if std.iloc[i]==1 and std.iloc[i-1]==-1: sc+=3
        if std.iloc[i]==-1 and std.iloc[i-1]==1: sc-=3
        if adx.iloc[i]>25 and pdi.iloc[i]>mdi.iloc[i]: sc+=1
        if adx.iloc[i]>25 and pdi.iloc[i]<mdi.iloc[i]: sc-=1
        if sc>=4: sigs.append({"t":df.index[i],"type":"LONG","y":df["low"].iloc[i]*0.9985})
        elif sc<=-4: sigs.append({"t":df.index[i],"type":"SHORT","y":df["high"].iloc[i]*1.0015})
    return sigs[-20:]

# ── RISK MANAGER ───────────────────────────────────────────────────────────

def risk_calc(price,atr_v,capital,risk_pct,direction,rr):
    ra=capital*risk_pct/100; sld=atr_v*1.5
    if direction=="LONG":
        sl=price-sld; tp1=price+sld*rr; tp2=price+sld*rr*1.618
    else:
        sl=price+sld; tp1=price-sld*rr; tp2=price-sld*rr*1.618
    size=ra/sld if sld>0 else 0
    return {"sl":sl,"tp1":tp1,"tp2":tp2,"size":size,
            "risk_usd":ra,"sl_pct":sld/price*100,"rr":rr}

# ── JARVIS ENGINE ──────────────────────────────────────────────────────────

class Jarvis:
    def __init__(self):
        self.fh=deque(maxlen=500); self.dh=deque(maxlen=500)
    def _f(self,df):
        rv=rsi14(df["close"]); tp=(df["high"]+df["low"]+df["close"])/3
        m=tp.rolling(20).mean(); md=tp.rolling(20).apply(lambda x:np.abs(x-x.mean()).mean(),raw=True)
        cc=(tp-m)/(0.015*md.replace(0,np.nan))
        return pd.DataFrame({"r":((rv.fillna(50)-30)/40).clip(0,1),
                             "c":((cc.fillna(0)+200)/400).clip(0,1)}).fillna(0.5)
    def knn(self,df,k=8):
        f=self._f(df).iloc[-1].values
        if len(self.fh)<k: return 0
        X=np.array(self.fh); y=np.array(self.dh)
        return 1 if y[np.argsort(np.sqrt(((f-X)**2).sum(axis=1)))[:k]].sum()>0 else -1
    def update(self,df,d): self.fh.append(self._f(df).iloc[-1].values); self.dh.append(d)
    def score(self,df,eps,htf_sc=50):
        sc=0; rs=[]; kd=self.knn(df)
        if kd!=0: sc+=15; rs.append("🧠 k-NN: +15")
        ev={p:df[f"EMA_{p}"].iloc[-1] for p in eps if f"EMA_{p}" in df.columns}; sp=sorted(ev)
        if len(sp)>=2:
            if all(ev[sp[i]]>ev[sp[i+1]] for i in range(len(sp)-1)): sc+=15; rs.append("📈 EMA bull: +15")
            elif all(ev[sp[i]]<ev[sp[i+1]] for i in range(len(sp)-1)): sc+=15; rs.append("📉 EMA bear: +15")
        if "MACD" in df.columns and df["MACD"].iloc[-1]>df["MACD_sig"].iloc[-1]: sc+=8; rs.append("📊 MACD bull: +8")
        if "RSI" in df.columns:
            rv=df["RSI"].iloc[-1]
            if rv<30: sc+=12; rs.append("📊 RSI oversold: +12")
            elif rv>70: sc+=12; rs.append("📊 RSI overbought: +12")
            elif 40<rv<60: sc+=4; rs.append("📊 RSI neutro: +4")
        if "ST_d" in df.columns:
            s=df["ST_d"].iloc[-1]
            if s==1: sc+=12; rs.append("🌊 SuperTrend ↑: +12")
            elif s==-1: sc+=12; rs.append("🌊 SuperTrend ↓: +12")
        if "ADX" in df.columns and df["ADX"].iloc[-1]>25: sc+=8; rs.append("💪 ADX>25: +8")
        hb=int((htf_sc-50)/8)
        if hb: sc+=hb; rs.append(f"📡 HTF: {hb:+d}")
        if "VOLMA" in df.columns and df["volume"].iloc[-1]>df["VOLMA"].iloc[-1]*1.5:
            sc+=5; rs.append("🔊 Vol spike: +5")
        return max(0,min(100,sc)),rs,kd
    def signal(self,df,eps,thr,htf_sc=50):
        sc,rs,kd=self.score(df,eps,htf_sc)
        if sc>=thr: sig="LONG" if (kd==1 or sc>=75) else ("SHORT" if kd==-1 else "NEUTRAL")
        else: sig="NEUTRAL"
        return {"signal":sig,"confidence":sc,"reasons":rs,"knn":kd}

# ── GRAFICO ────────────────────────────────────────────────────────────────

def make_chart(df,eps,overlays,osc_sel,signals,rk,direction,exchange,height):
    INC,DEC="#26a69a","#ef5350"
    cl=df["close"]; lo=df["low"]; hi=df["high"]

    # Calcolo range Y REALE — questo è il fix definitivo
    y_lo = float(lo.min()); y_hi = float(hi.max()); y_pad=(y_hi-y_lo)*0.05
    y_min = y_lo - y_pad; y_max = y_hi + y_pad

    # Oscillatori da mettere in subplot separati SOLO se selezionati
    sub_oscs=[(k,l,c) for k,l,c in [
        ("RSI","RSI","#ce93d8"),("MACD","MACD","#2196f3"),
        ("Stoch","Stoch","#4fc3f7"),("ADX","ADX","#ffeb3b"),
        ("CCI","CCI","#80cbc4")] if k in osc_sel]
    n_sub=len(sub_oscs)

    # Proporzioni righe: 1=prezzo grande, 2=volume piccolo, 3+=oscillatori
    if n_sub==0:
        row_h=[0.88,0.12]; n_rows=2
    else:
        osc_h=0.20/n_sub
        price_h=0.62; vol_h=0.08
        row_h=[price_h,vol_h]+[osc_h]*n_sub
        n_rows=2+n_sub

    fig=make_subplots(rows=n_rows,cols=1,shared_xaxes=True,
        vertical_spacing=0.02,row_heights=row_h,
        subplot_titles=(["",""]+ [l for _,l,_ in sub_oscs]) if n_sub else ["",""])

    # ── Candele ──
    fig.add_trace(go.Candlestick(
        x=df.index,open=df["open"],high=hi,low=lo,close=cl,name="Prezzo",
        increasing=dict(line=dict(color=INC,width=1),fillcolor=INC),
        decreasing=dict(line=dict(color=DEC,width=1),fillcolor=DEC),
        whiskerwidth=0.85),row=1,col=1)

    # ── Volume (barre basse proporzionate al range Y del prezzo) ──
    vol_max=df["volume"].max()
    vol_h_range=(y_hi-y_lo)*0.14
    vol_y=y_min+(df["volume"]/(vol_max if vol_max>0 else 1)*vol_h_range)
    vc=[INC if c>=o else DEC for c,o in zip(cl,df["open"])]
    fig.add_trace(go.Bar(x=df.index,y=vol_y-y_min,base=y_min,name="Vol",
        marker_color=vc,opacity=0.22,
        hovertemplate="Vol: %{customdata:,.0f}<extra></extra>",
        customdata=df["volume"]),row=1,col=1)

    # ── EMA ──
    EC={5:"#00e5ff",10:"#69f0ae",20:"#ffeb3b",50:"#ff9800",100:"#ce93d8",200:"#ef9a9a"}
    for p in eps:
        if f"EMA_{p}" in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[f"EMA_{p}"],name=f"EMA{p}",
                line=dict(color=EC.get(p,"#888"),width=1.6),opacity=0.9),row=1,col=1)

    # ── Bollinger ──
    if "BB" in overlays and "BB_up" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_up"],name="BB↑",
            line=dict(color="rgba(120,160,255,0.55)",width=1,dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_dn"],name="BB↓",
            line=dict(color="rgba(120,160,255,0.55)",width=1,dash="dot"),
            fill="tonexty",fillcolor="rgba(120,160,255,0.04)"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_mid"],name="BB mid",
            line=dict(color="rgba(200,200,200,0.18)",width=1)),row=1,col=1)

    # ── VWAP ──
    if "VWAP" in overlays and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["VWAP"],name="VWAP",
            line=dict(color="#ff6b6b",width=2,dash="dashdot")),row=1,col=1)

    # ── SuperTrend ──
    if "SuperTrend" in overlays and "ST" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["ST"].where(df["ST_d"]==1),
            name="ST↑",line=dict(color="#00e676",width=2.5),connectgaps=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["ST"].where(df["ST_d"]==-1),
            name="ST↓",line=dict(color="#ff1744",width=2.5),connectgaps=False),row=1,col=1)

    # ── Support / Resistance ──
    if "S/R" in overlays:
        res_lvls,sup_lvls=pivot_sr(df)
        for lv in res_lvls:
            if y_min<lv<y_max:
                fig.add_hline(y=lv,line_dash="dot",line_color="rgba(239,83,80,0.5)",
                    line_width=1.2,row=1,col=1,
                    annotation_text=f"R {lv:.5g}",
                    annotation_font_color="rgba(239,83,80,0.85)",
                    annotation_position="left")
        for lv in sup_lvls:
            if y_min<lv<y_max:
                fig.add_hline(y=lv,line_dash="dot",line_color="rgba(38,166,154,0.5)",
                    line_width=1.2,row=1,col=1,
                    annotation_text=f"S {lv:.5g}",
                    annotation_font_color="rgba(38,166,154,0.85)",
                    annotation_position="left")

    # ── Frecce segnali ──
    if signals:
        lx=[s["t"] for s in signals if s["type"]=="LONG"]
        ly=[s["y"] for s in signals if s["type"]=="LONG"]
        sx=[s["t"] for s in signals if s["type"]=="SHORT"]
        sy=[s["y"] for s in signals if s["type"]=="SHORT"]
        if lx: fig.add_trace(go.Scatter(x=lx,y=ly,mode="markers",name="▲ LONG",
            marker=dict(symbol="triangle-up",size=13,color="#00e676",
                line=dict(width=1,color="#003300"))),row=1,col=1)
        if sx: fig.add_trace(go.Scatter(x=sx,y=sy,mode="markers",name="▼ SHORT",
            marker=dict(symbol="triangle-down",size=13,color="#ff1744",
                line=dict(width=1,color="#330000"))),row=1,col=1)

    # ── SL / TP ──
    if rk and direction!="NEUTRAL":
        price=float(cl.iloc[-1])
        clr_sl="#ff1744"; clr_tp="#00e676"
        for lv,lbl,clr in [
            (rk["sl"],f"SL  {rk['sl']:.5g}  (-{rk['sl_pct']:.2f}%)",clr_sl),
            (rk["tp1"],f"TP1  {rk['tp1']:.5g}",clr_tp),
            (rk["tp2"],f"TP2  {rk['tp2']:.5g}",clr_tp),
        ]:
            if y_min*0.8<lv<y_max*1.2:
                fig.add_hline(y=lv,line_dash="solid" if "SL" in lbl else "dot",
                    line_color=clr,line_width=1.8,row=1,col=1,
                    annotation_text=lbl,annotation_font_color=clr,
                    annotation_position="right")
        lo_band=min(price,rk["sl"]); hi_band=max(price,rk["sl"])
        fig.add_hrect(y0=lo_band,y1=hi_band,fillcolor="rgba(239,83,80,0.06)",
            layer="below",line_width=0,row=1,col=1)
        lo_tp=min(price,rk["tp1"]); hi_tp=max(price,rk["tp1"])
        fig.add_hrect(y0=lo_tp,y1=hi_tp,fillcolor="rgba(38,166,154,0.06)",
            layer="below",line_width=0,row=1,col=1)

    # ── Volume subplot ──
    fig.add_trace(go.Bar(x=df.index,y=df["volume"],name="Vol",
        marker_color=vc,opacity=0.65,showlegend=False,
        hovertemplate="Vol: %{y:,.0f}<extra></extra>"),row=2,col=1)

    # ── Oscillatori subplot ──
    for idx_,(key,lbl,color) in enumerate(sub_oscs):
        r=3+idx_
        if key=="RSI" and "RSI" in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df["RSI"],name="RSI",
                line=dict(color=color,width=1.8),showlegend=False),row=r,col=1)
            for lv,c_ in [(70,"rgba(239,83,80,0.4)"),(50,"rgba(200,200,200,0.15)"),(30,"rgba(38,166,154,0.4)")]:
                fig.add_hline(y=lv,line_dash="dash",line_color=c_,line_width=1,row=r,col=1)
            fig.update_yaxes(range=[0,100],row=r,col=1)
        elif key=="MACD" and "MACD" in df.columns:
            hc=[INC if v>=0 else DEC for v in df["MACD_hist"]]
            fig.add_trace(go.Bar(x=df.index,y=df["MACD_hist"],marker_color=hc,
                opacity=0.7,name="Hist",showlegend=False),row=r,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["MACD"],name="MACD",
                line=dict(color=color,width=1.5),showlegend=False),row=r,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["MACD_sig"],name="Sig",
                line=dict(color="#ff9800",width=1.3,dash="dot"),showlegend=False),row=r,col=1)
            fig.add_hline(y=0,line_color="rgba(255,255,255,0.1)",line_width=1,row=r,col=1)
        elif key=="Stoch" and "SK" in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df["SK"],name="%K",
                line=dict(color=color,width=1.8),showlegend=False),row=r,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["SD"],name="%D",
                line=dict(color="#ff8a65",width=1.3,dash="dot"),showlegend=False),row=r,col=1)
            for lv,c_ in [(80,"rgba(239,83,80,0.4)"),(20,"rgba(38,166,154,0.4)")]:
                fig.add_hline(y=lv,line_dash="dash",line_color=c_,line_width=1,row=r,col=1)
            fig.update_yaxes(range=[0,100],row=r,col=1)
        elif key=="ADX" and "ADX" in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df["ADX"],name="ADX",
                line=dict(color=color,width=1.8),showlegend=False),row=r,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["DI_p"],name="DI+",
                line=dict(color="#69f0ae",width=1.2,dash="dot"),showlegend=False),row=r,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["DI_m"],name="DI-",
                line=dict(color="#ef9a9a",width=1.2,dash="dot"),showlegend=False),row=r,col=1)
            fig.add_hline(y=25,line_dash="dash",line_color="rgba(255,255,255,0.2)",
                line_width=1,row=r,col=1)
        elif key=="CCI" and "CCI" in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df["CCI"],name="CCI",
                line=dict(color=color,width=1.8),showlegend=False),row=r,col=1)
            for lv,c_ in [(100,"rgba(239,83,80,0.4)"),(0,"rgba(200,200,200,0.15)"),(-100,"rgba(38,166,154,0.4)")]:
                fig.add_hline(y=lv,line_dash="dash",line_color=c_,line_width=1,row=r,col=1)

    # ── Fix asse Y prezzo: forza range dai dati reali ──
    fig.update_yaxes(range=[y_min,y_max],row=1,col=1)

    ex_col=EX_CLR.get(exchange,"#fff")
    fig.update_layout(
        template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
        height=height,margin=dict(l=70,r=80,t=48,b=35),
        hovermode="x unified",xaxis_rangeslider_visible=False,
        title=dict(text=f'<span style="color:{ex_col};font-size:15px">● {exchange}</span>',
            x=0.01,font=dict(size=14,color="#c9d1d9")),
        showlegend=True,
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0,
            bgcolor="rgba(13,17,23,0.82)",bordercolor="#30363d",borderwidth=1,
            font=dict(size=11),itemsizing="constant"),
    )
    for i in range(1,n_rows+1):
        fig.update_xaxes(gridcolor="#1e2329",zeroline=False,
            rangeslider_visible=False,row=i,col=1)
        fig.update_yaxes(gridcolor="#1e2329",zeroline=False,row=i,col=1)
    return fig

# ── CALENDARIO ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600,show_spinner=False)
def get_calendar():
    try:
        r=requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=10,headers=HDR)
        if r.status_code==200:
            ev=[]
            for x in r.json():
                if not isinstance(x,dict): continue
                try: dt=datetime.strptime(x.get("date","")[:19],"%Y-%m-%dT%H:%M:%S"); ds=dt.strftime("%Y-%m-%d"); ts=dt.strftime("%H:%M")
                except: ds=str(datetime.now().date()); ts="00:00"
                ev.append({"date":ds,"time":ts,"country":x.get("country",""),
                    "event":x.get("title","N/A"),"impact":(x.get("impact") or "Low").upper(),
                    "prev":x.get("previous") or "-","forecast":x.get("forecast") or "-",
                    "actual":x.get("actual") or ""})
            return ev
    except: pass
    today=datetime.now()
    return [{"date":(today+timedelta(days=d)).strftime("%Y-%m-%d"),"time":"14:30",
             "country":c,"event":n,"impact":i,"prev":p,"forecast":f,"actual":""}
            for d,n,c,i,p,f in [(1,"Non-Farm Payrolls","🇺🇸","HIGH","175K","180K"),
                (3,"CPI Inflation","🇺🇸","HIGH","3.2%","3.1%"),(5,"FOMC","🇺🇸","HIGH","5.25%","5.25%"),
                (7,"ECB Rate","🇪🇺","HIGH","4.00%","4.00%"),(14,"Core PCE","🇺🇸","MEDIUM","2.7%","2.6%")]]


# ── TOP 100 CRIPTO PER CAP ─────────────────────────────────────────────────
TOP20 = [
    "BTC","ETH","BNB","SOL","XRP","DOGE","ADA","AVAX","TRX","SHIB",
    "DOT","LINK","MATIC","LTC","UNI","ATOM","XLM","APT","ICP","OP"
]
TOP80 = [
    "ARB","FIL","HBAR","VET","INJ","IMX","MNT","GRT","STX","NEAR",
    "AAVE","SAND","MANA","ENS","LDO","RPL","CRV","SNX","BAL","COMP",
    "1INCH","SUSHI","YFI","MKR","DYDX","RUNE","KAVA","FTM","ONE","ZIL",
    "QTUM","ONT","ICX","ZRX","BAT","KNC","STORJ","OGN","REN","BAND",
    "OCEAN","FET","AGI","NMR","ANKR","COTI","CELR","SKL","CHZ","FLOW",
    "ROSE","ALPHA","BETA","TWT","DENT","HOT","WIN","BTT","NFT","POLS",
    "AUCTION","KEEP","NU","PERP","MDT","LINA","VITE","STMX","TROY","IRIS",
    "DOCK","EASY","FOR","DF","BNT","TRB","REP","MLN","RSR","OXT"
]
TOP100 = TOP20 + TOP80

# ── TRADE ZONE ANALYZER ───────────────────────────────────────────────────

def analyze_trade_zone(df, coin, tf):
    """Analisi in tempo reale: Trade Zone o No-Trade Zone"""
    if df.empty or len(df) < 50:
        return {"zone": "UNKNOWN", "score": 0, "reasons": [], "color": "gray"}

    cl = df["close"]; hi = df["high"]; lo = df["low"]
    reasons = []; score = 0

    # 1. Trend chiarezza — EMA stack
    e20 = ema(cl, 20).iloc[-1]; e50 = ema(cl, 50).iloc[-1]
    e200 = ema(cl, 200).iloc[-1]; px = float(cl.iloc[-1])
    if px > e20 > e50 > e200:
        score += 20; reasons.append("✅ EMA stack bullish perfetto")
    elif px < e20 < e50 < e200:
        score += 20; reasons.append("✅ EMA stack bearish perfetto")
    elif (px > e20 and e20 > e50) or (px < e20 and e20 < e50):
        score += 10; reasons.append("⚠️ EMA parzialmente allineate")
    else:
        score -= 10; reasons.append("❌ EMA in conflitto — range laterale")

    # 2. ADX — forza del trend
    adx_s, _, _ = adx_full(df)
    adx_v = float(adx_s.iloc[-1])
    if adx_v >= 30:
        score += 25; reasons.append(f"✅ ADX {adx_v:.0f} — trend forte")
    elif adx_v >= 22:
        score += 12; reasons.append(f"⚠️ ADX {adx_v:.0f} — trend moderato")
    else:
        score -= 15; reasons.append(f"❌ ADX {adx_v:.0f} — mercato laterale")

    # 3. SuperTrend conferma
    _, std = supertrend(df)
    if std.iloc[-1] == std.iloc[-3]:  # stabile, non appena invertito
        score += 15; reasons.append("✅ SuperTrend stabile — direzionale")
    else:
        score -= 5; reasons.append("⚠️ SuperTrend appena invertito — attenzione")

    # 4. ATR volatilità sufficiente
    atr_v = float(atr14(df).iloc[-1])
    atr_pct = atr_v / px * 100
    if atr_pct >= 0.8:
        score += 15; reasons.append(f"✅ ATR {atr_pct:.2f}% — volatilità buona")
    elif atr_pct >= 0.4:
        score += 5; reasons.append(f"⚠️ ATR {atr_pct:.2f}% — volatilità bassa")
    else:
        score -= 10; reasons.append(f"❌ ATR {atr_pct:.2f}% — mercato piatto")

    # 5. RSI non in zona di esaurimento estremo
    rv = float(rsi14(cl).iloc[-1])
    if 35 < rv < 65:
        score += 10; reasons.append(f"✅ RSI {rv:.0f} — zona neutrale operabile")
    elif rv <= 20 or rv >= 80:
        score -= 15; reasons.append(f"❌ RSI {rv:.0f} — estremo, rischio inversione")
    else:
        score += 5; reasons.append(f"⚠️ RSI {rv:.0f} — zona di attenzione")

    # 6. Volume > media
    vol_ma = float(df["volume"].rolling(20).mean().iloc[-1])
    vol_cur = float(df["volume"].iloc[-1])
    if vol_cur > vol_ma * 1.3:
        score += 15; reasons.append("✅ Volume sopra media — partecipazione alta")
    elif vol_cur > vol_ma * 0.8:
        score += 5; reasons.append("⚠️ Volume nella norma")
    else:
        score -= 10; reasons.append("❌ Volume basso — scarsa partecipazione")

    score = max(0, min(100, score))

    if score >= 65:
        zone = "TRADE ZONE"
        color = "green"
        emoji = "🟢"
    elif score >= 40:
        zone = "ZONA ATTENZIONE"
        color = "orange"
        emoji = "🟡"
    else:
        zone = "NO-TRADE ZONE"
        color = "red"
        emoji = "🔴"

    return {"zone": zone, "score": score, "reasons": reasons,
            "color": color, "emoji": emoji,
            "adx": adx_v, "atr_pct": atr_pct, "rsi": rv}


def render_trade_zone(tz):
    """Rende il box Trade Zone sopra i segnali"""
    clr_map = {"green": "#003300", "orange": "#332200", "red": "#330000"}
    brd_map  = {"green": "#00e676", "orange": "#ffb300", "red": "#ef5350"}
    txt_map  = {"green": "#00e676", "orange": "#ffb300", "red": "#ef5350"}
    bg   = clr_map.get(tz["color"], "#1a1a1a")
    brd  = brd_map.get(tz["color"], "#555")
    txt  = txt_map.get(tz["color"], "#ccc")

    reasons_html = "".join(
        f'<div style="font-size:12px;color:#c9d1d9;margin:2px 0">{r}</div>'
        for r in tz["reasons"]
    )
    st.markdown(f"""
    <div style="background:{bg};border:2px solid {brd};border-radius:10px;
         padding:14px 18px;margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:22px;font-weight:700;color:{txt}">
          {tz["emoji"]} {tz["zone"]}
        </span>
        <span style="font-size:28px;font-weight:800;color:{txt}">
          {tz["score"]}/100
        </span>
      </div>
      <div style="margin-top:4px;font-size:12px;color:#8b949e">
        ADX {tz["adx"]:.0f} &nbsp;|&nbsp; ATR {tz["atr_pct"]:.2f}% &nbsp;|&nbsp; RSI {tz["rsi"]:.0f}
      </div>
      <div style="margin-top:10px;border-top:1px solid #30363d;padding-top:8px">
        {reasons_html}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── SCANNER TOP 100 ────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def scanner_group(symbols, tf_scan="1h"):
    results = []
    for sym in symbols:
        pair_b = f"{sym}USDT"
        df_s, _ = get_ohlcv({"exchange":"Binance","symbol":sym,"pair":pair_b}, tf_scan, 150)
        if df_s.empty or len(df_s) < 50:
            continue
        df_s = add_all(df_s, [20, 50])
        cl = df_s["close"]
        curr = float(cl.iloc[-1])
        atr_v = float(df_s["ATR"].iloc[-1]) if "ATR" in df_s.columns else curr * 0.01

        sc = 0
        e20 = float(df_s["EMA_20"].iloc[-1]); e50 = float(df_s["EMA_50"].iloc[-1])
        if curr > e20 > e50:   sc += 20
        elif curr < e20 < e50: sc -= 20
        if df_s["MACD"].iloc[-1] > df_s["MACD_sig"].iloc[-1]: sc += 15
        else: sc -= 15
        rv = float(df_s["RSI"].iloc[-1])
        if rv < 35:   sc += 18
        elif rv > 65: sc -= 18
        elif rv > 50: sc += 5
        else:         sc -= 5
        std = float(df_s["ST_d"].iloc[-1]) if "ST_d" in df_s.columns else 0
        if std ==  1: sc += 20
        elif std == -1: sc -= 20
        adx_v = float(df_s["ADX"].iloc[-1]) if "ADX" in df_s.columns else 0
        if adx_v > 25: sc = int(sc * 1.15)
        if float(df_s["volume"].iloc[-1]) > float(df_s["VOLMA"].iloc[-1]) * 1.4:
            sc = int(sc * 1.1)

        sc_norm = max(0, min(100, 50 + sc))
        if sc_norm >= 65:     direction = "LONG"
        elif sc_norm <= 35:   direction = "SHORT"
        else: continue

        sl_dist = atr_v * 1.5
        if direction == "LONG":
            sl  = curr - sl_dist; tp1 = curr + sl_dist * 2.0; tp2 = curr + sl_dist * 3.236
        else:
            sl  = curr + sl_dist; tp1 = curr - sl_dist * 2.0; tp2 = curr - sl_dist * 3.236

        chg = (curr / float(cl.iloc[-2]) - 1) * 100 if float(cl.iloc[-2]) else 0
        results.append({"sym":sym,"direction":direction,"score":sc_norm,
            "price":curr,"sl":sl,"tp1":tp1,"tp2":tp2,"rsi":rv,"adx":adx_v,"chg":chg})

    results.sort(key=lambda x: abs(x["score"] - 50), reverse=True)
    return results


def _signal_card(s):
    """Renderizza una card segnale"""
    is_long = s["direction"] == "LONG"
    clr  = "#00e676" if is_long else "#ef5350"
    arrow = "^" if is_long else "v"
    px = s["price"]
    fmt = f"${px:,.6f}" if px < 0.01 else f"${px:,.4f}" if px < 1 else f"${px:,.2f}"
    chg_col = "#00e676" if s["chg"] >= 0 else "#ef5350"
    bg_lbl  = "#0d2e1a" if is_long else "#2e0d0d"

    st.markdown(f"""
    <div style="background:#161b22;border:1px solid #30363d;
         border-left:4px solid {clr};border-radius:8px;
         padding:10px 12px;margin-bottom:8px;font-size:13px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="color:{clr};font-weight:700;font-size:15px">{arrow} {s["sym"]}/USDT</span>
        <span style="color:{clr};background:{bg_lbl};padding:2px 9px;
              border-radius:11px;font-weight:600;font-size:12px">
          {"LONG" if is_long else "SHORT"} {s["score"]}/100</span>
      </div>
      <div style="color:#8b949e;margin:4px 0;font-size:11px">
        <span style="color:{chg_col}">{s["chg"]:+.2f}%</span>
        &nbsp;·&nbsp; RSI {s["rsi"]:.0f} &nbsp;·&nbsp; ADX {s["adx"]:.0f}
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
           gap:3px;text-align:center;margin-top:6px">
        <div style="background:#0d1117;border-radius:4px;padding:4px">
          <div style="color:#8b949e;font-size:10px">ENTRY</div>
          <div style="color:#ffeb3b;font-weight:600;font-size:11px">{fmt}</div>
        </div>
        <div style="background:#0d1117;border-radius:4px;padding:4px">
          <div style="color:#ef5350;font-size:10px">SL</div>
          <div style="color:#ef5350;font-weight:600;font-size:11px">${s["sl"]:,.5g}</div>
        </div>
        <div style="background:#0d1117;border-radius:4px;padding:4px">
          <div style="color:#00e676;font-size:10px">TP1</div>
          <div style="color:#00e676;font-weight:600;font-size:11px">${s["tp1"]:,.5g}</div>
        </div>
        <div style="background:#0d1117;border-radius:4px;padding:4px">
          <div style="color:#69f0ae;font-size:10px">TP2</div>
          <div style="color:#69f0ae;font-weight:600;font-size:11px">${s["tp2"]:,.5g}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_scanner_full(container, tf):
    with container:
        st.subheader("🔭 Scanner Top 100")
        st.caption(f"Motore Jarvis · TF: {tf} · cache 2 min")

        tab_big, tab_alt = st.tabs(["🏆 Big 20", "🚀 Alt 80"])

        with tab_big:
            col_scan_btn = st.columns([3,1])
            with col_scan_btn[1]:
                if st.button("🔍 Scan Big 20", key="scan_big", use_container_width=True):
                    st.cache_data.clear()
            with st.spinner("Scansione Big 20..."):
                big_sigs = scanner_group(TOP20, tf)
            longs  = [s for s in big_sigs if s["direction"]=="LONG"]
            shorts = [s for s in big_sigs if s["direction"]=="SHORT"]
            if not big_sigs:
                st.info("Nessun segnale chiaro al momento.")
            else:
                st.markdown(f"**🟢 {len(longs)} LONG &nbsp; 🔴 {len(shorts)} SHORT**")
                st.divider()
                for s in big_sigs:
                    _signal_card(s)

        with tab_alt:
            col_scan_btn2 = st.columns([3,1])
            with col_scan_btn2[1]:
                if st.button("🔍 Scan Alt 80", key="scan_alt", use_container_width=True):
                    st.cache_data.clear()
            with st.spinner("Scansione Alt 80..."):
                alt_sigs = scanner_group(TOP80, tf)
            longs2  = [s for s in alt_sigs if s["direction"]=="LONG"]
            shorts2 = [s for s in alt_sigs if s["direction"]=="SHORT"]
            if not alt_sigs:
                st.info("Nessun segnale chiaro al momento.")
            else:
                st.markdown(f"**🟢 {len(longs2)} LONG &nbsp; 🔴 {len(shorts2)} SHORT**")
                st.divider()
                for s in alt_sigs:
                    _signal_card(s)


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    st.markdown("""<style>
    [data-testid="stMetric"]{background:#161b22;border-radius:8px;
        padding:10px 14px;border:1px solid #30363d;}
    </style>""", unsafe_allow_html=True)
    st.title("🧠 Jarvis Pro — Crypto Trading Assistant")
    st.caption("🔵 Binance · 🟠 KuCoin · 🟣 OKX · Multi-TF · Risk Manager · Scanner Top 100")

    if "jarvis" not in st.session_state:
        st.session_state.jarvis = Jarvis()
    jarvis = st.session_state.jarvis

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Setup")
        with st.spinner("📡 Caricamento mercati..."):
            coins = get_all_coins()
        st.caption(f"✅ {len(coins)} coin disponibili")
        srch = st.text_input("🔍 Cerca", "")
        filt = [c for c in coins if srch.upper() in c["symbol"]] if srch else coins
        sel  = st.selectbox("💎 Coin", [c["display"] for c in filt[:500]] if filt else ["—"])
        coin = next((c for c in filt if c["display"] == sel), coins[0] if coins else None)
        if not coin:
            st.error("Nessun coin"); return
        ex = coin["exchange"]
        st.info(f"**{EX_ICONS.get(ex,'⚪')} {ex}** · `{coin['pair']}`")
        tf_opts = TF_BIN if ex=="Binance" else list(TF_KUC.keys()) if ex=="KuCoin" else list(TF_OKX.keys())
        tf = st.selectbox("⏱️ Timeframe", tf_opts,
            index=tf_opts.index("1h") if "1h" in tf_opts else 0)
        limit = st.slider("📦 Candele", 100, 500, 250, step=50)
        st.divider()
        st.subheader("📊 Indicatori")
        ema_s = st.multiselect("EMA",
            ["EMA_5","EMA_10","EMA_20","EMA_50","EMA_100","EMA_200"],
            default=["EMA_20","EMA_50","EMA_200"])
        eps  = [int(e.split("_")[1]) for e in ema_s] or [50]
        ovl  = st.multiselect("Overlay",
            ["BB","VWAP","SuperTrend","S/R"], default=["BB","SuperTrend","S/R"])
        osc  = st.multiselect("Oscillatori",
            ["RSI","MACD","Stoch","ADX","CCI"], default=["RSI","MACD"])
        show_sig = st.checkbox("▲▼ Frecce segnali", value=True)
        st.divider()
        st.subheader("💰 Risk Manager")
        capital = st.number_input("Capitale ($)", 100, 500000, 1000, step=500)
        risk_p  = st.slider("Rischio %", 0.5, 5.0, 1.0, step=0.5)
        rr_r    = st.slider("Risk:Reward", 1.0, 5.0, 2.0, step=0.5)
        st.divider()
        soglia  = st.slider("🎯 Soglia AI", 40, 80, 60)
        h_chart = st.slider("📐 Altezza grafico", 600, 1200, 820, step=50)
        if st.button("🔄 Aggiorna", type="primary", use_container_width=True):
            st.cache_data.clear(); st.rerun()

    # ── Layout: col principale | col destra ──────────────────────────────
    col_main, col_right = st.columns([3, 1])

    # ── Caricamento dati asset selezionato ───────────────────────────────
    with st.spinner(f"📡 {coin['symbol']}..."):
        df_raw, src = get_ohlcv(coin, tf, limit)
    if df_raw.empty:
        st.error("❌ Nessun dato. Prova altro timeframe."); return
    df = add_all(df_raw, eps)
    with st.spinner("📡 HTF..."):
        htf_sc, htf_nm = get_htf(coin, tf)
    res  = jarvis.signal(df, eps, soglia, htf_sc)
    if len(df) > 2:
        jarvis.update(df, 1 if df["close"].iloc[-1] > df["close"].iloc[-2] else -1)

    sig  = res["signal"]; sc = res["confidence"]
    ic   = EX_ICONS.get(src, "⚪")
    curr = float(df["close"].iloc[-1]); prev = float(df["close"].iloc[-2])
    dpct = (curr / prev - 1) * 100 if prev else 0
    atr_v = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else curr * 0.01
    rk   = risk_calc(curr, atr_v, capital, risk_p, sig, rr_r) if sig != "NEUTRAL" else None

    # ── Colonna principale ────────────────────────────────────────────────
    with col_main:
        htf_str = f"HTF {htf_sc}/100 ({htf_nm})"
        if sig == "LONG":    st.success(f"🟢 **LONG — {sc}/100**  ·  {ic} {src}  ·  {tf}  ·  {htf_str}")
        elif sig == "SHORT": st.error(  f"🔴 **SHORT — {sc}/100**  ·  {ic} {src}  ·  {tf}  ·  {htf_str}")
        else:                st.warning(f"⚪ **NEUTRAL — {sc}/100**  ·  {ic} {src}  ·  {tf}  ·  {htf_str}")

        px_fmt = f"${curr:,.6f}" if curr<0.01 else f"${curr:,.4f}" if curr<1 else f"${curr:,.2f}"
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("💰 Prezzo",   px_fmt, f"{dpct:+.2f}%")
        c2.metric("📉 RSI",      f"{df['RSI'].iloc[-1]:.1f}" if "RSI" in df.columns else "—")
        c3.metric("💪 ADX",      f"{df['ADX'].iloc[-1]:.1f}" if "ADX" in df.columns else "—")
        std_v = df["ST_d"].iloc[-1] if "ST_d" in df.columns else 0
        c4.metric("🌊 SuperTrend","🟢 BULL" if std_v==1 else "🔴 BEAR")
        c5.metric("📡 HTF",      f"{htf_sc}/100", htf_nm)
        c6.metric("📐 ATR",      f"{atr_v:.5g}")

        if rk and sig != "NEUTRAL":
            st.markdown("---")
            st.subheader(f"💰 Piano trade — {'🟢 LONG' if sig=='LONG' else '🔴 SHORT'}")
            r1,r2,r3,r4,r5 = st.columns(5)
            r1.metric("🎯 Entry",          f"${curr:,.5g}")
            r2.metric("🛑 Stop Loss",      f"${rk['sl']:,.5g}",  f"-{rk['sl_pct']:.2f}%")
            r3.metric(f"✅ TP1 (1:{rr_r:.1f})", f"${rk['tp1']:,.5g}")
            r4.metric("🚀 TP2 (ext.)",     f"${rk['tp2']:,.5g}")
            r5.metric("⚠️ Rischio $",      f"${rk['risk_usd']:.2f}", f"{risk_p}%")
            sz     = f"{rk['size']:.6f}" if curr<1 else f"{rk['size']:.4f}"
            valore = rk["size"] * curr
            st.info(f"📦 **Size:** `{sz}` unità · **Valore:** `${valore:,.2f}` · **R:R 1:{rr_r:.1f}`")
            st.markdown("---")

        with st.expander("🧠 AI Score — confluenza dettagliata"):
            ca, cb = st.columns(2)
            ca.write(f"**LTF score:** {sc}/100")
            ca.write(f"**HTF score ({htf_nm}):** {htf_sc}/100")
            ca.write(f"**k-NN:** {'🟢 LONG' if res['knn']==1 else '🔴 SHORT' if res['knn']==-1 else '⚪'}")
            for r_ in res["reasons"]: cb.write(f"• {r_}")

        sigs = detect_signals(df, eps) if show_sig else []
        fig  = make_chart(df, eps, ovl, osc, sigs, rk, sig, src, h_chart)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🖱️ Clicca legenda per attivare/disattivare · Scroll = zoom · Drag = pan")

        with st.expander("📋 Tabella dati"):
            cols_ = ["open","high","low","close","volume","RSI","MACD","ADX","SK","ATR","CCI"]
            st.dataframe(df.tail(30)[[c for c in cols_ if c in df.columns]].round(6),
                use_container_width=True)

    # ── Colonna destra ────────────────────────────────────────────────────
    with col_right:

        # 1. TRADE ZONE ANALYZER — asset selezionato (IN PRIMO PIANO)
        st.subheader("🎯 Trade Zone Analyzer")
        st.caption(f"{coin['symbol']} · {tf} · analisi live")
        with st.spinner("Analisi in corso..."):
            tz = analyze_trade_zone(df, coin, tf)
        render_trade_zone(tz)

        st.divider()

        # 2. SCANNER TOP 100 — sotto la trade zone
        render_scanner_full(st.container(), tf)

        st.divider()

        # 3. CALENDARIO ECONOMICO — in fondo
        st.subheader("📅 Calendario Economico")
        with st.spinner("..."): ev_all = get_calendar()
        sd  = st.date_input("Da", datetime.now().date())
        ed  = st.date_input("A",  datetime.now().date() + timedelta(days=14))
        imp_f = st.multiselect("Impatto", ["HIGH","MEDIUM","LOW"], default=["HIGH","MEDIUM"])
        evs = sorted([e for e in ev_all
                      if imp_f and e.get("date")
                      and sd <= datetime.strptime(e["date"],"%Y-%m-%d").date() <= ed
                      and e["impact"] in imp_f],
                     key=lambda x: x["date"] + x.get("time",""))
        for e in evs:
            ic2 = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(e["impact"],"⚪")
            act = f" → **{e['actual']}**" if e.get("actual") else ""
            st.markdown(f"{ic2} **{e['event']}**{act}")
            st.caption(f"{e['country']} · {e['date']} {e.get('time','')}")
            st.caption(f"Prec:`{e['prev']}` · Prev:`{e['forecast']}`")
            st.divider()
        if not evs: st.info("Nessun evento")

main()
