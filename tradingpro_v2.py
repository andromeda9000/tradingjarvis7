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

st.set_page_config(page_title="Jarvis V7 - Crypto", page_icon="🧠", layout="wide")

# ─────────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────────
BIN = "https://api.binance.com/api/v3"
KUC = "https://api.kucoin.com/api/v1"
OKX = "https://www.okx.com/api/v5"
HDR = {"User-Agent": "Mozilla/5.0"}

# Mapping timeframe per exchange
TF_BIN = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w"]
TF_KUC = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min",
           "1h":"1hour","2h":"2hour","4h":"4hour","6h":"6hour","8h":"8hour",
           "12h":"12hour","1d":"1day","1w":"1week"}
TF_OKX = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
           "1h":"1H","2h":"2H","4h":"4H","6h":"6H","8h":"8H",
           "12h":"12H","1d":"1D","1w":"1W"}

EXCHANGE_COLORS = {
    "Binance": "🔵",
    "KuCoin":  "🟠",
    "OKX":     "🟣",
}

# ─────────────────────────────────────────────
# LISTA COIN DA TUTTI E TRE GLI EXCHANGE
# ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_binance_tickers():
    try:
        r = requests.get(f"{BIN}/ticker/24hr", timeout=12)
        if r.status_code != 200: return []
        data = r.json()
        if not isinstance(data, list): return []
        out = []
        for item in data:
            if not isinstance(item, dict): continue
            sym = item.get("symbol","")
            if not sym.endswith("USDT"): continue
            base = sym[:-4]
            px   = float(item.get("lastPrice",0) or 0)
            chg  = float(item.get("priceChangePercent",0) or 0)
            vol  = float(item.get("quoteVolume",0) or 0)
            out.append({"exchange":"Binance","symbol":base,"pair":sym,
                        "price":px,"chg":chg,"vol":vol})
        return sorted(out, key=lambda x:-x["vol"])
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def get_kucoin_tickers():
    try:
        r = requests.get(f"{KUC}/market/allTickers", timeout=12)
        if r.status_code != 200: return []
        data = r.json().get("data",{}).get("ticker",[])
        if not isinstance(data, list): return []
        out = []
        for item in data:
            if not isinstance(item, dict): continue
            sym = item.get("symbol","")
            if not sym.endswith("-USDT"): continue
            base = sym[:-5]
            px   = float(item.get("last",0) or 0)
            chg  = float(item.get("changeRate",0) or 0) * 100
            vol  = float(item.get("volValue",0) or 0)
            out.append({"exchange":"KuCoin","symbol":base,"pair":sym,
                        "price":px,"chg":chg,"vol":vol})
        return sorted(out, key=lambda x:-x["vol"])
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def get_okx_tickers():
    try:
        r = requests.get(f"{OKX}/market/tickers", params={"instType":"SPOT"}, timeout=12)
        if r.status_code != 200: return []
        data = r.json().get("data",[])
        if not isinstance(data, list): return []
        out = []
        for item in data:
            if not isinstance(item, dict): continue
            sym = item.get("instId","")
            if not sym.endswith("-USDT"): continue
            base = sym[:-5]
            px   = float(item.get("last",0) or 0)
            open24 = float(item.get("open24h",px) or px)
            chg  = ((px/open24)-1)*100 if open24 else 0
            vol  = float(item.get("volCcy24h",0) or 0)
            out.append({"exchange":"OKX","symbol":base,"pair":sym,
                        "price":px,"chg":chg,"vol":vol})
        return sorted(out, key=lambda x:-x["vol"])
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def get_all_coins():
    bin_list = get_binance_tickers()
    kuc_list = get_kucoin_tickers()
    okx_list = get_okx_tickers()

    # Unisci dando priorità a Binance → KuCoin → OKX
    # Per ogni simbolo, mantieni la fonte con volume più alto
    seen = {}
    for coin in bin_list + kuc_list + okx_list:
        sym = coin["symbol"]
        if sym not in seen or coin["vol"] > seen[sym]["vol"]:
            seen[sym] = coin

    result = []
    for coin in sorted(seen.values(), key=lambda x:-x["vol"]):
        ex  = coin["exchange"]
        arr = "🟢" if coin["chg"] >= 0 else "🔴"
        clr = EXCHANGE_COLORS.get(ex, "⚪")
        px  = coin["price"]
        fmt = f"${px:,.6f}" if px < 0.01 else f"${px:,.4f}" if px < 1 else f"${px:,.2f}"
        coin["display"] = f"{clr} {arr} {coin['symbol']} — {fmt}  ({coin['chg']:+.2f}%)  [{ex}]"
        result.append(coin)
    return result

# ─────────────────────────────────────────────
# DATI OHLCV
# ─────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def ohlcv_binance(pair, tf, limit=300):
    try:
        r = requests.get(f"{BIN}/klines", timeout=10,
            params={"symbol":pair,"interval":tf,"limit":limit})
        if r.status_code != 200: return pd.DataFrame()
        data = r.json()
        if not isinstance(data, list) or not data: return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            "ts","open","high","low","close","volume",
            "ct","qv","n","tbb","tbq","ign"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df.dropna(subset=["open","high","low","close"])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def ohlcv_kucoin(pair, tf, limit=300):
    try:
        kf = TF_KUC.get(tf)
        if not kf: return pd.DataFrame()
        end_ts = int(time.time())
        # KuCoin klines: startAt, endAt in secondi
        r = requests.get(f"{KUC}/market/candles", timeout=10,
            params={"symbol":pair,"type":kf,"endAt":end_ts,"startAt":end_ts-limit*_tf_secs(tf)})
        if r.status_code != 200: return pd.DataFrame()
        data = r.json().get("data",[])
        if not isinstance(data, list) or not data: return pd.DataFrame()
        df = pd.DataFrame(data, columns=["ts","open","close","high","low","volume","amount"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="s")
        df.set_index("ts", inplace=True)
        df.sort_index(inplace=True)
        return df.dropna(subset=["open","high","low","close"])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def ohlcv_okx(pair, tf, limit=300):
    try:
        okf = TF_OKX.get(tf)
        if not okf: return pd.DataFrame()
        r = requests.get(f"{OKX}/market/history-candles", timeout=10,
            params={"instId":pair,"bar":okf,"limit":min(limit,300)})
        if r.status_code != 200: return pd.DataFrame()
        data = r.json().get("data",[])
        if not isinstance(data, list) or not data: return pd.DataFrame()
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
        for c in ["open","high","low","close","vol"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.rename(columns={"vol":"volume"}, inplace=True)
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        df.set_index("ts", inplace=True)
        df.sort_index(inplace=True)
        return df.dropna(subset=["open","high","low","close"])
    except Exception:
        return pd.DataFrame()

def _tf_secs(tf):
    mp = {"1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,
          "1h":3600,"2h":7200,"4h":14400,"6h":21600,"8h":28800,
          "12h":43200,"1d":86400,"3d":259200,"1w":604800}
    return mp.get(tf, 3600)

def get_ohlcv(coin, tf, limit):
    ex = coin.get("exchange")
    pair = coin.get("pair","")
    if ex == "Binance":
        df = ohlcv_binance(pair, tf, limit)
        if not df.empty: return df, "Binance"
    elif ex == "KuCoin":
        df = ohlcv_kucoin(pair, tf, limit)
        if not df.empty: return df, "KuCoin"
    elif ex == "OKX":
        df = ohlcv_okx(pair, tf, limit)
        if not df.empty: return df, "OKX"
    # Fallback: prova gli altri exchange
    for try_ex, try_fn, try_pair in [
        ("Binance", ohlcv_binance, f"{coin['symbol']}USDT"),
        ("KuCoin",  ohlcv_kucoin,  f"{coin['symbol']}-USDT"),
        ("OKX",     ohlcv_okx,     f"{coin['symbol']}-USDT"),
    ]:
        if try_ex == ex: continue
        df = try_fn(try_pair, tf, limit)
        if not df.empty:
            return df, try_ex
    return pd.DataFrame(), "—"

# ─────────────────────────────────────────────
# INDICATORI TECNICI
# ─────────────────────────────────────────────

def ema(s,p):   return s.ewm(span=p,adjust=False).mean()
def rsi(s,p=14):
    d=s.diff(); g=d.clip(lower=0).rolling(p).mean()
    l=(-d.clip(upper=0)).rolling(p).mean()
    return (100-(100/(1+g/l.replace(0,np.nan)))).fillna(50)

def macd(s):
    m=ema(s,12)-ema(s,26); sig=m.ewm(span=9,adjust=False).mean()
    return m, sig, m-sig

def bollinger(s,p=20,k=2):
    m=s.rolling(p).mean(); sd=s.rolling(p).std()
    return m+sd*k, m-sd*k, m

def atr(df,p=14):
    h,l,c=df["high"],df["low"],df["close"]
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def adx_calc(df,p=14):
    h,l,c=df["high"].values,df["low"].values,df["close"].values
    n=len(df); pdm=np.zeros(n); mdm=np.zeros(n)
    for i in range(1,n):
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        if up>dn and up>0: pdm[i]=up
        if dn>up and dn>0: mdm[i]=dn
    tr_a=np.zeros(n)
    for i in range(1,n):
        tr_a[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    def rma(a,per):
        o=np.zeros(n); al=1/per
        if per<n: o[per]=np.mean(a[:per+1])
        for i in range(per+1,n): o[i]=o[i-1]*(1-al)+a[i]*al
        return o
    at=rma(tr_a,p)
    pdi=np.where(at>0,100*rma(pdm,p)/at,0)
    mdi=np.where(at>0,100*rma(mdm,p)/at,0)
    sm=pdi+mdi; dx=np.where(sm>0,100*np.abs(pdi-mdi)/sm,0)
    idx=df.index
    return (pd.Series(rma(dx,p),index=idx,dtype=float),
            pd.Series(pdi,index=idx,dtype=float),
            pd.Series(mdi,index=idx,dtype=float))

def stoch(df,k=14,d=3):
    lo=df["low"].rolling(k).min(); hi=df["high"].rolling(k).max()
    K=100*(df["close"]-lo)/(hi-lo).replace(0,np.nan)
    return K.fillna(50), K.rolling(d).mean().fillna(50)

def vwap_calc(df):
    tp=(df["high"]+df["low"]+df["close"])/3
    vol=df["volume"].replace(0,np.nan)
    return (tp*vol).cumsum()/vol.cumsum()

def supertrend_calc(df,p=10,m=3.0):
    at=atr(df,p).values; hl=((df["high"]+df["low"])/2).values; cl=df["close"].values
    n=len(df); ub=hl+m*at; lb=hl-m*at
    fub,flb=ub.copy(),lb.copy()
    d=np.ones(n,dtype=int); sv=np.full(n,np.nan)
    for i in range(1,n):
        fub[i]=min(ub[i],fub[i-1]) if cl[i-1]<=fub[i-1] else ub[i]
        flb[i]=max(lb[i],flb[i-1]) if cl[i-1]>=flb[i-1] else lb[i]
        if cl[i]>fub[i-1]: d[i]=1
        elif cl[i]<flb[i-1]: d[i]=-1
        else: d[i]=d[i-1]
        sv[i]=flb[i] if d[i]==1 else fub[i]
    idx=df.index
    return pd.Series(sv,index=idx,dtype=float), pd.Series(d.astype(float),index=idx)

def cci_calc(df,p=20):
    tp=(df["high"]+df["low"]+df["close"])/3
    m=tp.rolling(p).mean()
    md=tp.rolling(p).apply(lambda x:np.abs(x-x.mean()).mean(),raw=True)
    return ((tp-m)/(0.015*md.replace(0,np.nan))).fillna(0)

def add_indicators(df, eps):
    df=df.copy(); cl=df["close"]
    for p in eps: df[f"EMA_{p}"]=ema(cl,p)
    df["RSI"]=rsi(cl)
    df["MACD_line"],df["MACD_signal"],df["MACD_hist"]=macd(cl)
    df["BB_upper"],df["BB_lower"],df["BB_mid"]=bollinger(cl)
    df["ATR"]=atr(df)
    df["ADX"],df["DI_plus"],df["DI_minus"]=adx_calc(df)
    df["STOCH_K"],df["STOCH_D"]=stoch(df)
    df["VWAP"]=vwap_calc(df)
    df["ST"],df["ST_dir"]=supertrend_calc(df)
    df["CCI"]=cci_calc(df)
    return df

# ─────────────────────────────────────────────
# JARVIS ENGINE
# ─────────────────────────────────────────────

class Jarvis:
    def __init__(self):
        self.fh=deque(maxlen=500); self.dh=deque(maxlen=500)

    def _feats(self,df):
        r=rsi(df["close"]); tp=(df["high"]+df["low"]+df["close"])/3
        m=tp.rolling(20).mean()
        md=tp.rolling(20).apply(lambda x:np.abs(x-x.mean()).mean(),raw=True)
        c=(tp-m)/(0.015*md.replace(0,np.nan))
        return pd.DataFrame({"r":((r.fillna(50)-30)/40).clip(0,1),
                              "c":((c.fillna(0)+200)/400).clip(0,1)}).fillna(0.5)

    def knn(self,df,k=8):
        f=self._feats(df).iloc[-1].values
        if len(self.fh)<k: return 0
        X=np.array(self.fh); y=np.array(self.dh)
        dists=np.sqrt(((f-X)**2).sum(axis=1))
        return 1 if y[np.argsort(dists)[:k]].sum()>0 else -1

    def update(self,df,d):
        self.fh.append(self._feats(df).iloc[-1].values); self.dh.append(d)

    def regime(self,df):
        if "ADX" not in df.columns or len(df)<50: return 1.0
        ap=(atr(df)/df["close"].replace(0,np.nan))*100
        q=ap.rolling(50).quantile(0.33).iloc[-1]
        if not np.isfinite(q): return 1.0
        if ap.iloc[-1]<q: return 0.85
        adv=df["ADX"].iloc[-1]
        if np.isfinite(adv) and adv>50: return 0.60
        return 1.0

    def score(self,df,eps):
        sc=0; rs=[]; kd=self.knn(df)
        if kd!=0: sc+=20; rs.append(f"🧠 k-NN: +20")
        ev={p:df[f"EMA_{p}"].iloc[-1] for p in eps if f"EMA_{p}" in df.columns}
        sp=sorted(ev)
        if len(sp)>=2:
            if all(ev[sp[i]]>ev[sp[i+1]] for i in range(len(sp)-1)): sc+=20; rs.append("📈 EMA bull: +20")
            elif all(ev[sp[i]]<ev[sp[i+1]] for i in range(len(sp)-1)): sc+=20; rs.append("📉 EMA bear: +20")
        if "MACD_line" in df.columns and df["MACD_line"].iloc[-1]>df["MACD_signal"].iloc[-1]: sc+=5; rs.append("📊 MACD: +5")
        if "RSI" in df.columns:
            rv=df["RSI"].iloc[-1]
            if 40<rv<60: sc+=5; rs.append("📊 RSI neutro: +5")
            elif rv<30: sc+=10; rs.append("📊 RSI oversold: +10")
            elif rv>70: sc+=10; rs.append("📊 RSI overbought: +10")
        if "ST_dir" in df.columns:
            s=df["ST_dir"].iloc[-1]
            if s==1: sc+=10; rs.append("🌊 SuperTrend bull: +10")
            elif s==-1: sc+=10; rs.append("🌊 SuperTrend bear: +10")
        if "ADX" in df.columns and df["ADX"].iloc[-1]>25: sc+=5; rs.append("💪 Trend forte: +5")
        mult=self.regime(df); orig=sc; sc=int(min(sc*mult,100))
        if mult<1: rs.append(f"⚡ Regime: x{mult:.2f} ({orig}→{sc})")
        return sc, rs, kd

    def signal(self,df,eps,thr=60):
        sc,rs,kd=self.score(df,eps); mult=self.regime(df)
        if sc>=thr and mult>=0.7:
            sig="LONG" if kd==1 or sc>=75 else ("SHORT" if kd==-1 else "NEUTRAL")
        else: sig="NEUTRAL"
        return {"signal":sig,"confidence":sc,"reasons":rs,"knn":kd,"regime":mult}

# ─────────────────────────────────────────────
# GRAFICO
# ─────────────────────────────────────────────

EX_CHART_COLORS = {
    "Binance": "#00b4d8",  # azzurro
    "KuCoin":  "#f77f00",  # arancio
    "OKX":     "#a855f7",  # viola
}

def make_chart(df, eps, show, src_label, exchange):
    INC,DEC="#26a69a","#ef5350"
    ex_color = EX_CHART_COLORS.get(exchange, "#ffffff")

    rows=[(1,0.45),(2,0.10)]
    smap={}
    for key,lbl,flag in [
        ("macd","⚡ MACD","MACD" in show),
        ("rsi","📉 RSI","RSI" in show),
        ("stoch","🔄 Stoch","Stoch" in show),
        ("adx","💪 ADX","ADX" in show),
        ("cci","📐 CCI","CCI" in show),
    ]:
        if flag:
            smap[key]=len(rows)+1
            rows.append((len(rows)+1,0.13))

    nr=len(rows); tot=sum(r[1] for r in rows)
    fig=make_subplots(rows=nr,cols=1,shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[r[1]/tot for r in rows],
        subplot_titles=["📈 Prezzo","📊 Volume"]+
            [{"macd":"⚡ MACD","rsi":"📉 RSI","stoch":"🔄 Stoch",
              "adx":"💪 ADX","cci":"📐 CCI"}[k] for k in smap])

    # Candele con bordo exchange color
    fig.add_trace(go.Candlestick(x=df.index,
        open=df["open"],high=df["high"],low=df["low"],close=df["close"],
        name="Prezzo",increasing_line_color=INC,decreasing_line_color=DEC,
        increasing_fillcolor=INC,decreasing_fillcolor=DEC),row=1,col=1)

    # Linea exchange color sotto il titolo
    fig.add_hrect(y0=0,y1=0, line_width=0, row=1, col=1)

    EC={5:"#00e5ff",10:"#69f0ae",20:"#ffeb3b",50:"#ff9800",100:"#ce93d8",200:"#ef9a9a"}
    for p in eps:
        cn=f"EMA_{p}"
        if cn in show and cn in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[cn],name=f"EMA{p}",
                line=dict(color=EC.get(p,"#aaa"),width=1.5)),row=1,col=1)

    if "BB" in show and "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_upper"],name="BB↑",
            line=dict(color="rgba(100,150,255,0.7)",width=1,dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_lower"],name="BB↓",
            line=dict(color="rgba(100,150,255,0.7)",width=1,dash="dot"),
            fill="tonexty",fillcolor="rgba(100,150,255,0.04)"),row=1,col=1)

    if "VWAP" in show and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["VWAP"],name="VWAP",
            line=dict(color="#ff6b6b",width=2,dash="dashdot")),row=1,col=1)

    if "SuperTrend" in show and "ST" in df.columns:
        bull=df["ST"].where(df["ST_dir"]==1)
        bear=df["ST"].where(df["ST_dir"]==-1)
        fig.add_trace(go.Scatter(x=df.index,y=bull,name="ST↑",
            line=dict(color="#00e676",width=2),connectgaps=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=bear,name="ST↓",
            line=dict(color="#ff1744",width=2),connectgaps=False),row=1,col=1)

    vc=[INC if c>=o else DEC for c,o in zip(df["close"],df["open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["volume"],name="Vol",
        marker_color=vc,opacity=0.6),row=2,col=1)

    if "macd" in smap and "MACD_line" in df.columns:
        r=smap["macd"]
        hc=[INC if v>=0 else DEC for v in df["MACD_hist"]]
        fig.add_trace(go.Bar(x=df.index,y=df["MACD_hist"],marker_color=hc,opacity=0.7,name="Hist"),row=r,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["MACD_line"],name="MACD",line=dict(color="#2196f3",width=1.5)),row=r,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["MACD_signal"],name="Sig",line=dict(color="#ff9800",width=1.5)),row=r,col=1)

    if "rsi" in smap and "RSI" in df.columns:
        r=smap["rsi"]
        fig.add_trace(go.Scatter(x=df.index,y=df["RSI"],name="RSI",
            line=dict(color="#ce93d8",width=2)),row=r,col=1)
        for lv,cl_ in [(70,DEC),(50,"gray"),(30,INC)]:
            fig.add_hline(y=lv,line_dash="dash",line_color=cl_,line_width=1,row=r,col=1)
        fig.update_yaxes(range=[0,100],row=r,col=1)

    if "stoch" in smap and "STOCH_K" in df.columns:
        r=smap["stoch"]
        fig.add_trace(go.Scatter(x=df.index,y=df["STOCH_K"],name="%K",line=dict(color="#4fc3f7",width=1.5)),row=r,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["STOCH_D"],name="%D",line=dict(color="#ff8a65",width=1.5)),row=r,col=1)
        for lv,cl_ in [(80,DEC),(20,INC)]:
            fig.add_hline(y=lv,line_dash="dash",line_color=cl_,line_width=1,row=r,col=1)
        fig.update_yaxes(range=[0,100],row=r,col=1)

    if "adx" in smap and "ADX" in df.columns:
        r=smap["adx"]
        fig.add_trace(go.Scatter(x=df.index,y=df["ADX"],name="ADX",line=dict(color="#ffeb3b",width=2)),row=r,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["DI_plus"],name="DI+",line=dict(color="#69f0ae",width=1.2,dash="dot")),row=r,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["DI_minus"],name="DI-",line=dict(color="#ef9a9a",width=1.2,dash="dot")),row=r,col=1)
        fig.add_hline(y=25,line_dash="dash",line_color="gray",line_width=1,row=r,col=1)

    if "cci" in smap and "CCI" in df.columns:
        r=smap["cci"]
        fig.add_trace(go.Scatter(x=df.index,y=df["CCI"],name="CCI",line=dict(color="#80cbc4",width=1.5)),row=r,col=1)
        for lv,cl_ in [(100,DEC),(0,"gray"),(-100,INC)]:
            fig.add_hline(y=lv,line_dash="dash",line_color=cl_,line_width=1,row=r,col=1)

    fig.update_layout(
        title=dict(
            text=f'<span style="color:{ex_color}">●</span>  {src_label}',
            font=dict(size=15,color="#e0e0e0"),x=0.01),
        template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#161b22",
        height=max(850, 420 + nr * 120),showlegend=True,
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,
            bgcolor="rgba(13,17,23,0.8)",bordercolor="#30363d",borderwidth=1,
            font=dict(size=10)),
        hovermode="x unified",margin=dict(l=60,r=20,t=70,b=40))
    for i in range(1,nr+1):
        fig.update_xaxes(gridcolor="#21262d",zeroline=False,rangeslider_visible=False,row=i,col=1)
        fig.update_yaxes(gridcolor="#21262d",zeroline=False,row=i,col=1)
    return fig

# ─────────────────────────────────────────────
# CALENDARIO ECONOMICO
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_calendar():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=10, headers=HDR)
        if r.status_code == 200:
            raw = r.json()
            ev = []
            for item in raw:
                if not isinstance(item, dict): continue
                raw_d = item.get("date","")
                try: dt=datetime.strptime(raw_d[:19],"%Y-%m-%dT%H:%M:%S"); ds=dt.strftime("%Y-%m-%d"); ts=dt.strftime("%H:%M")
                except: ds=str(datetime.now().date()); ts="00:00"
                ev.append({"date":ds,"time":ts,
                    "country":item.get("country",""),
                    "event":item.get("title","N/A"),
                    "impact":(item.get("impact") or "Low").upper(),
                    "prev":item.get("previous") or "-",
                    "forecast":item.get("forecast") or "-",
                    "actual":item.get("actual") or ""})
            return ev
    except Exception:
        pass
    today=datetime.now()
    fb=[
        (1,"Non-Farm Payrolls","🇺🇸","HIGH","175K","180K"),
        (3,"CPI Inflation","🇺🇸","HIGH","3.2%","3.1%"),
        (5,"FOMC Minutes","🇺🇸","HIGH","5.25%","5.25%"),
        (7,"ECB Rate Decision","🇪🇺","HIGH","4.00%","4.00%"),
        (10,"GDP Q1","🇺🇸","HIGH","2.1%","2.3%"),
        (14,"Core PCE","🇺🇸","MEDIUM","2.7%","2.6%"),
        (16,"Retail Sales","🇺🇸","MEDIUM","0.7%","0.5%"),
        (18,"Jobless Claims","🇺🇸","LOW","220K","215K"),
    ]
    return [{"date":(today+timedelta(days=d)).strftime("%Y-%m-%d"),"time":"14:30",
             "country":c,"event":n,"impact":i,"prev":p,"forecast":f,"actual":""}
            for d,n,c,i,p,f in fb]

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    st.markdown("""<style>
    [data-testid="stMetric"]{background:#161b22;border-radius:8px;padding:10px 14px;border:1px solid #30363d;}
    </style>""", unsafe_allow_html=True)

    st.title("🧠 Jarvis V7 — Crypto Dashboard")
    st.caption(
        "🔵 **Binance**  ·  "
        "🟠 **KuCoin**  ·  "
        "🟣 **OKX**  ·  "
        "SMC · AI Signal · Calendario"
    )

    if "jarvis" not in st.session_state:
        st.session_state.jarvis = Jarvis()
    jarvis = st.session_state.jarvis

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.header("⚙️ Impostazioni")

        with st.spinner("📡 Caricamento coin da Binance, KuCoin, OKX..."):
            all_coins = get_all_coins()

        st.caption(f"✅ {len(all_coins)} coin disponibili")

        search = st.text_input("🔍 Cerca coin", "")
        if search:
            s = search.upper()
            filtered = [c for c in all_coins if s in c["symbol"]]
        else:
            filtered = all_coins

        sel = st.selectbox("💎 Seleziona Coin",
            [c["display"] for c in filtered[:500]] if filtered else ["—"])
        coin = next((c for c in filtered if c["display"]==sel), all_coins[0] if all_coins else None)

        if coin is None:
            st.error("Nessun coin disponibile."); return

        ex = coin["exchange"]
        ex_icon = EXCHANGE_COLORS.get(ex,"⚪")
        st.info(f"**{ex_icon} {ex}**  ·  {coin['symbol']}/USDT  ·  Pair: `{coin['pair']}`")

        # Timeframe disponibili per exchange
        if ex == "KuCoin":
            tf_opts = list(TF_KUC.keys())
        elif ex == "OKX":
            tf_opts = list(TF_OKX.keys())
        else:
            tf_opts = TF_BIN

        tf_idx = tf_opts.index("1h") if "1h" in tf_opts else 0
        tf = st.selectbox("⏱️ Timeframe", tf_opts, index=tf_idx)
        limit = st.slider("📦 Candele", 100, 500, 250, step=50)

        st.divider()
        st.subheader("📊 Indicatori")
        ema_sel = st.multiselect("EMA",
            ["EMA_5","EMA_10","EMA_20","EMA_50","EMA_100","EMA_200"],
            default=["EMA_20","EMA_50","EMA_200"])
        eps = [int(e.split("_")[1]) for e in ema_sel] or [50]

        overlay = st.multiselect("Overlay",
            ["BB","VWAP","SuperTrend"], default=["BB","SuperTrend"])
        subs = st.multiselect("Oscillatori",
            ["MACD","RSI","Stoch","ADX","CCI"], default=["MACD","RSI","ADX"])
        show = set(ema_sel + overlay + subs)

        st.divider()
        soglia = st.slider("🎯 Soglia AI", 40, 80, 60)
        if st.button("🔄 Aggiorna", type="primary", use_container_width=True):
            st.cache_data.clear(); st.rerun()

    # ── LAYOUT PRINCIPALE ────────────────────
    col_main, col_cal = st.columns([3,1])

    with col_main:
        with st.spinner(f"📡 Download {coin['symbol']} da {ex}..."):
            df_raw, src_used = get_ohlcv(coin, tf, limit)

        if df_raw.empty:
            st.error(f"❌ Nessun dato per **{coin['symbol']}** ({ex}). Prova altro timeframe o exchange.")
            return

        df = add_indicators(df_raw, eps)
        res = jarvis.signal(df, eps, soglia)
        if len(df) > 2:
            jarvis.update(df, 1 if df["close"].iloc[-1]>df["close"].iloc[-2] else -1)

        sig=res["signal"]; sc=res["confidence"]
        ex_icon=EXCHANGE_COLORS.get(src_used,"⚪")
        label=f"{ex_icon} **{src_used}** · {coin['symbol']}/USDT · {tf}"

        if   sig=="LONG":  st.success(f"🟢 **LONG — AI {sc}/100**  ·  {ex_icon} {src_used} · {tf}")
        elif sig=="SHORT": st.error(  f"🔴 **SHORT — AI {sc}/100**  ·  {ex_icon} {src_used} · {tf}")
        else:              st.warning(f"⚪ **NEUTRAL — AI {sc}/100**  ·  {ex_icon} {src_used} · {tf}")

        curr=df["close"].iloc[-1]; prev=df["close"].iloc[-2]
        dpct=(curr/prev-1)*100 if prev else 0
        c1,c2,c3,c4,c5,c6=st.columns(6)
        px_fmt = f"${curr:,.6f}" if curr < 0.01 else f"${curr:,.4f}" if curr < 1 else f"${curr:,.2f}"
        c1.metric("💰 Prezzo", px_fmt, f"{dpct:+.2f}%")
        c2.metric("📉 RSI",    f"{df['RSI'].iloc[-1]:.1f}"   if "RSI" in df.columns else "—")
        c3.metric("💪 ADX",    f"{df['ADX'].iloc[-1]:.1f}"   if "ADX" in df.columns else "—")
        c4.metric("📊 ATR",    f"{df['ATR'].iloc[-1]:.5g}"   if "ATR" in df.columns else "—")
        st_v=df["ST_dir"].iloc[-1] if "ST_dir" in df.columns else 0
        c5.metric("🌊 ST",     "🟢 BULL" if st_v==1 else "🔴 BEAR")
        cci_v=df["CCI"].iloc[-1] if "CCI" in df.columns else 0
        c6.metric("📐 CCI",    f"{cci_v:.0f}")

        with st.expander("🧠 AI Reasons"):
            ca,cb=st.columns(2)
            ca.write(f"**Score:** {sc}/100  ·  **Regime:** x{res['regime']:.2f}")
            for r_ in res["reasons"]: cb.write(f"• {r_}")

        fig=make_chart(df, eps, show,
            f"{coin['symbol']}/USDT  ·  {src_used}  ·  {tf}", src_used)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Dati recenti"):
            dcols=["open","high","low","close","volume","RSI","MACD_line","ADX","STOCH_K","ATR","CCI"]
            st.dataframe(df.tail(20)[[c for c in dcols if c in df.columns]].round(6),
                use_container_width=True)

    with col_cal:
        st.subheader("📅 Calendario Economico")
        with st.spinner("..."):
            ev_all = get_calendar()
        sd=st.date_input("Da", datetime.now().date())
        ed=st.date_input("A",  datetime.now().date()+timedelta(days=14))
        imp_f=st.multiselect("Impatto",["HIGH","MEDIUM","LOW"],default=["HIGH","MEDIUM"])
        evs=sorted([e for e in ev_all
                    if imp_f and e.get("date")
                    and sd<=datetime.strptime(e["date"],"%Y-%m-%d").date()<=ed
                    and e["impact"] in imp_f],
                   key=lambda x:x["date"]+x.get("time",""))
        if evs:
            for e in evs:
                ic={"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(e["impact"],"⚪")
                act=f" → **{e['actual']}**" if e.get("actual") else ""
                st.markdown(f"{ic} **{e['event']}**{act}")
                st.caption(f"{e['country']} · {e['date']} {e.get('time','')}")
                st.caption(f"Prec: `{e['prev']}` · Prev: `{e['forecast']}`")
                st.divider()
        else:
            st.info("Nessun evento nel periodo")
        st.caption("🔴 HIGH = alta volatilità attesa")

main()
