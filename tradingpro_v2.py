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
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Jarvis V7 - Crypto Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 1. LISTA CRIPTO — CoinGecko (fonte primaria, tutti i coin)
# ============================================================================

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
BINANCE_BASE   = "https://api.binance.com/api/v3"

@st.cache_data(ttl=300)
def get_binance_symbols():
    """Recupera tutti i simboli USDT attivi su Binance per il routing."""
    try:
        r = requests.get(f"{BINANCE_BASE}/exchangeInfo", timeout=10)
        if r.status_code == 200:
            data = r.json()
            return {s["symbol"] for s in data.get("symbols", [])
                    if s.get("status") == "TRADING" and s["symbol"].endswith("USDT")}
    except Exception:
        pass
    return set()

@st.cache_data(ttl=300)
def get_coingecko_coins(pages=3):
    """Recupera fino a 750 coin da CoinGecko ordinati per market cap."""
    coins = []
    for page in range(1, pages + 1):
        try:
            r = requests.get(
                f"{COINGECKO_BASE}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": page,
                    "sparkline": "false",
                    "price_change_percentage": "24h",
                },
                timeout=12,
            )
            if r.status_code == 429:
                time.sleep(15)
                r = requests.get(r.url, timeout=12)
            if r.status_code != 200:
                break
            batch = r.json()
            if not isinstance(batch, list):
                break
            coins.extend(batch)
            if len(batch) < 250:
                break
        except Exception:
            break
    return coins


def build_crypto_list():
    """Costruisce la lista completa: CoinGecko + flag Binance disponibile."""
    binance_syms = get_binance_symbols()
    cg_coins     = get_coingecko_coins(pages=3)
    result = []
    for c in cg_coins:
        if not isinstance(c, dict):
            continue
        sym    = (c.get("symbol") or "").upper()
        name   = c.get("name", sym)
        cg_id  = c.get("id", "")
        price  = c.get("current_price") or 0
        chg    = c.get("price_change_percentage_24h") or 0
        vol    = c.get("total_volume") or 0
        mcap   = c.get("market_cap_rank") or 9999
        b_sym  = f"{sym}USDT"
        on_bin = b_sym in binance_syms
        arrow  = "🟢" if chg >= 0 else "🔴"
        src    = "🔵" if on_bin else "🟣"   # 🔵 Binance  🟣 CoinGecko
        result.append({
            "display":     f"{src} {arrow} {sym} — {name} ${price:,.4f} ({chg:+.2f}%)",
            "symbol":      sym,
            "name":        name,
            "cg_id":       cg_id,
            "binance_sym": b_sym if on_bin else None,
            "price":       price,
            "change":      chg,
            "volume":      vol,
            "rank":        mcap,
            "on_binance":  on_bin,
        })
    return result


# ============================================================================
# 2. DATI OHLCV — Binance (alta risoluzione) + CoinGecko fallback
# ============================================================================

# Mapping timeframe Binance → (giorni CoinGecko, label granularità CG)
TF_TO_CG = {
    "1m":  (1,   "~30 min"),
    "5m":  (1,   "~30 min"),
    "15m": (1,   "~30 min"),
    "30m": (1,   "~30 min"),
    "1h":  (1,   "~30 min"),
    "2h":  (7,   "~4 h"),
    "4h":  (7,   "~4 h"),
    "6h":  (14,  "~4 h"),
    "8h":  (30,  "~4 h"),
    "12h": (30,  "~4 h"),
    "1d":  (90,  "~4 gg"),
    "3d":  (180, "~4 gg"),
    "1w":  (365, "~4 gg"),
}

@st.cache_data(ttl=60)
def get_binance_ohlcv(symbol: str, interval: str, limit: int = 300):
    try:
        r = requests.get(f"{BINANCE_BASE}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            "ts","open","high","low","close","volume",
            "ct","qv","n","tbb","tbq","ignore"])
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        df.dropna(subset=["open","high","low","close"], inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_coingecko_ohlcv(cg_id: str, days: int):
    """OHLC da CoinGecko + volume da market_chart (merge per timestamp)."""
    try:
        # OHLC
        r_ohlc = requests.get(f"{COINGECKO_BASE}/coins/{cg_id}/ohlc",
            params={"vs_currency": "usd", "days": str(days)}, timeout=12)
        if r_ohlc.status_code == 429:
            time.sleep(12)
            r_ohlc = requests.get(r_ohlc.url, timeout=12)
        if r_ohlc.status_code != 200:
            return pd.DataFrame()
        ohlc = r_ohlc.json()
        if not isinstance(ohlc, list) or not ohlc:
            return pd.DataFrame()

        df = pd.DataFrame(ohlc, columns=["ts","open","high","low","close"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)

        # Volume
        try:
            time.sleep(2)
            r_vol = requests.get(f"{COINGECKO_BASE}/coins/{cg_id}/market_chart",
                params={"vs_currency":"usd","days":str(days)}, timeout=12)
            if r_vol.status_code == 200:
                vols = r_vol.json().get("total_volumes", [])
                df_v = pd.DataFrame(vols, columns=["ts","volume"])
                df_v["ts"] = pd.to_datetime(df_v["ts"], unit="ms")
                df_v.set_index("ts", inplace=True)
                df = df.join(df_v, how="left")
            else:
                df["volume"] = 0.0
        except Exception:
            df["volume"] = 0.0

        df.fillna(method="ffill", inplace=True)
        df.dropna(subset=["open","high","low","close"], inplace=True)
        return df

    except Exception as e:
        st.error(f"Errore CoinGecko OHLC: {e}")
        return pd.DataFrame()


def get_ohlcv(coin_info: dict, timeframe: str, limit: int):
    """Router: Binance se disponibile, altrimenti CoinGecko."""
    b_sym = coin_info.get("binance_sym")
    if b_sym:
        df = get_binance_ohlcv(b_sym, timeframe, limit)
        if not df.empty:
            return df, "Binance", timeframe
    # CoinGecko fallback
    cg_id = coin_info.get("cg_id")
    days, gran = TF_TO_CG.get(timeframe, (30, "~4 h"))
    if cg_id:
        df = get_coingecko_ohlcv(cg_id, days)
        if not df.empty:
            return df, "CoinGecko", gran
    return pd.DataFrame(), "—", "—"


# ============================================================================
# 3. CALENDARIO ECONOMICO
# ============================================================================

@st.cache_data(ttl=3600)
def get_economic_calendar():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            data = r.json()
            events = []
            for item in data:
                raw = item.get("date", "")
                try:
                    dt = datetime.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S")
                    d_str, t_str = dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
                except Exception:
                    d_str = raw[:10] if len(raw) >= 10 else str(datetime.now().date())
                    t_str = "00:00"
                events.append({
                    "date":     d_str,
                    "time":     t_str,
                    "country":  item.get("country", ""),
                    "event":    item.get("title", "N/A"),
                    "impact":   (item.get("impact") or "Low").upper(),
                    "prev":     item.get("previous") or "-",
                    "forecast": item.get("forecast") or "-",
                    "actual":   item.get("actual") or "",
                })
            return events
    except Exception:
        pass
    today = datetime.now()
    fb = [
        (1,  "Non-Farm Payrolls",      "🇺🇸","HIGH",  "175K","180K"),
        (3,  "CPI Inflation Rate",     "🇺🇸","HIGH",  "3.2%","3.1%"),
        (5,  "FOMC Minutes",           "🇺🇸","HIGH",  "5.25%","5.25%"),
        (7,  "ECB Rate Decision",      "🇪🇺","HIGH",  "4.00%","4.00%"),
        (10, "GDP Growth Rate Q1",     "🇺🇸","HIGH",  "2.1%","2.3%"),
        (12, "Bank of England Rate",   "🇬🇧","HIGH",  "5.00%","5.00%"),
        (14, "Core PCE Price Index",   "🇺🇸","MEDIUM","2.7%","2.6%"),
        (16, "Retail Sales m/m",       "🇺🇸","MEDIUM","0.7%","0.5%"),
        (18, "Initial Jobless Claims", "🇺🇸","LOW",   "220K","215K"),
        (20, "Flash PMI Manufacturing","🇪🇺","MEDIUM","47.3","47.8"),
    ]
    return [{"date":(today+timedelta(days=d)).strftime("%Y-%m-%d"),"time":"14:30",
             "country":c,"event":n,"impact":i,"prev":p,"forecast":f,"actual":""}
            for d,n,c,i,p,f in fb]


# ============================================================================
# 4. INDICATORI TECNICI
# ============================================================================

def _s(arr, idx): return pd.Series(arr, index=idx, dtype=float)

def ema(series, p):    return series.ewm(span=p, adjust=False).mean()
def rsi(series, p=14):
    d = series.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return (100-(100/(1+g/l.replace(0,np.nan)))).fillna(50)

def macd(series, f=12, s=26, sig=9):
    m = ema(series,f)-ema(series,s)
    sl = m.ewm(span=sig,adjust=False).mean()
    return m, sl, m-sl

def bollinger(series, p=20, k=2):
    sm = series.rolling(p).mean()
    sd = series.rolling(p).std()
    return sm+sd*k, sm-sd*k, sm

def atr(df, p=14):
    h,l,c = df["high"],df["low"],df["close"]
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def adx(df, p=14):
    h,l,c = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1,n):
        up = h[i]-h[i-1]; dn = l[i-1]-l[i]
        if up>dn and up>0: pdm[i]=up
        if dn>up and dn>0: mdm[i]=dn
    tr_a = np.concatenate([[0],np.maximum.reduce([h[1:]-l[1:],np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])])])
    def rma(a,per):
        o=np.zeros(n); o[per]=a[:per+1].mean(); al=1/per
        for i in range(per+1,n): o[i]=o[i-1]*(1-al)+a[i]*al
        return o
    at=rma(tr_a,p); pdi=100*rma(pdm,p)/np.where(at==0,np.nan,at)
    mdi=100*rma(mdm,p)/np.where(at==0,np.nan,at)
    dx=100*np.abs(pdi-mdi)/np.where((pdi+mdi)==0,np.nan,pdi+mdi)
    idx=df.index
    return _s(rma(np.nan_to_num(dx),p),idx), _s(pdi,idx), _s(mdi,idx)

def stochastic(df, kp=14, dp=3):
    lm=df['low'].rolling(kp).min(); hm=df['high'].rolling(kp).max()
    k=100*(df['close']-lm)/(hm-lm).replace(0,np.nan)
    return k.fillna(50), k.rolling(dp).mean().fillna(50)

def vwap(df):
    tp=(df['high']+df['low']+df['close'])/3
    return (tp*df['volume']).cumsum()/df['volume'].replace(0,np.nan).cumsum()

def supertrend(df, p=10, mult=3.0):
    at=atr(df,p).values; hl=((df['high']+df['low'])/2).values; cl=df['close'].values; n=len(df)
    ub=hl+mult*at; lb=hl-mult*at
    fub,flb=ub.copy(),lb.copy()
    d=np.ones(n,dtype=int); sv=np.full(n,np.nan)
    for i in range(1,n):
        fub[i]=min(ub[i],fub[i-1]) if cl[i-1]<=fub[i-1] else ub[i]
        flb[i]=max(lb[i],flb[i-1]) if cl[i-1]>=flb[i-1] else lb[i]
        if   cl[i]>fub[i-1]: d[i]=1
        elif cl[i]<flb[i-1]: d[i]=-1
        else: d[i]=d[i-1]
        sv[i]=flb[i] if d[i]==1 else fub[i]
    return _s(sv,df.index), _s(d.astype(float),df.index)

def cci(df, p=20):
    tp=(df['high']+df['low']+df['close'])/3
    sm=tp.rolling(p).mean()
    md=tp.rolling(p).apply(lambda x:np.abs(x-x.mean()).mean(),raw=True)
    return ((tp-sm)/(0.015*md.replace(0,np.nan))).fillna(0)

def add_indicators(df, ema_periods):
    df=df.copy(); cl=df["close"]
    for p in ema_periods: df[f"EMA_{p}"]=ema(cl,p)
    df["RSI"]=rsi(cl)
    df["MACD_line"],df["MACD_signal"],df["MACD_hist"]=macd(cl)
    df["BB_upper"],df["BB_lower"],df["BB_middle"]=bollinger(cl)
    df["ATR"]=atr(df)
    df["ADX"],df["DI_plus"],df["DI_minus"]=adx(df)
    df["STOCH_K"],df["STOCH_D"]=stochastic(df)
    df["VWAP"]=vwap(df)
    df["SUPERTREND"],df["SUPERTREND_DIR"]=supertrend(df)
    df["CCI"]=cci(df)
    return df


# ============================================================================
# 5. MOTORE JARVIS V7
# ============================================================================

class JarvisV7Engine:
    def __init__(self, hs=500, k=8):
        self.k=k
        self.fh=deque(maxlen=hs); self.dh=deque(maxlen=hs); self.wh=deque(maxlen=100)

    def _wma(self,s,p): return s.rolling(p).apply(lambda x:np.average(x,weights=np.arange(1,len(x)+1)),raw=True)
    def _hma(self,s,p):
        h,sq=max(int(p/2),1),max(int(np.sqrt(p)),1)
        return self._wma(2*self._wma(s,h)-self._wma(s,p),sq)

    def features(self,df):
        cl=df["close"]
        r=rsi(cl); tp=(df["high"]+df["low"]+df["close"])/3
        sm=tp.rolling(20).mean(); md=tp.rolling(20).apply(lambda x:np.abs(x-x.mean()).mean(),raw=True)
        c=(tp-sm)/(0.015*md.replace(0,np.nan))
        return pd.DataFrame({"rn":((self._hma(r.fillna(50),9)-30)/40).clip(0,1),
                              "cn":((self._hma(c.fillna(0),9)+200)/400).clip(0,1)}).fillna(0.5)

    def knn(self,df):
        cur=self.features(df).iloc[-1].values
        if len(self.fh)<self.k: return 0
        X=np.array(self.fh); y=np.array(self.dh)
        dists=np.sqrt(((np.log1p(cur+1e-6)-np.log1p(X+1e-6))**2).sum(axis=1))
        return 1 if y[np.argsort(dists)[:self.k]].sum()>0 else -1

    def update(self,df,d): self.fh.append(self.features(df).iloc[-1].values); self.dh.append(d)

    def regime(self,df):
        if "ADX" not in df.columns or len(df)<50: return 1.0
        ap=(atr(df)/df["close"].replace(0,np.nan))*100
        q=ap.rolling(50).quantile(0.33).iloc[-1]
        if not np.isfinite(q): return 1.0
        if ap.iloc[-1]<q: return 0.85
        adv=df["ADX"].iloc[-1]
        if np.isfinite(adv) and adv>50: return 0.60
        return 1.0

    def obs(self,df):
        h,l,c=df['high'],df['low'],df['close']; ob=[]; lb=min(20,len(df)-2)
        for i in range(lb,len(df)-1):
            if i<2: continue
            rng=h.iloc[i]-l.iloc[i]
            if rng<=0: continue
            if l.iloc[i]<l.iloc[i-1] and c.iloc[i]>c.iloc[i-2]:
                ob.append({'t':'bull','p':l.iloc[i],'s':min(abs(c.iloc[i]-l.iloc[i])/rng,1)})
            elif h.iloc[i]>h.iloc[i-1] and c.iloc[i]<c.iloc[i-2]:
                ob.append({'t':'bear','p':h.iloc[i],'s':min(abs(h.iloc[i]-c.iloc[i])/rng,1)})
        return ob[-5:] if ob else []

    def sweeps(self,df):
        if len(df)<50: return []
        w=min(20,len(df)//3); ph=[]; pl=[]
        for i in range(w,len(df)-w):
            if df['high'].iloc[i]>df['high'].iloc[i-w:i].max() and df['high'].iloc[i]>df['high'].iloc[i+1:i+w+1].max(): ph.append(df['high'].iloc[i])
            if df['low'].iloc[i]<df['low'].iloc[i-w:i].min() and df['low'].iloc[i]<df['low'].iloc[i+1:i+w+1].min(): pl.append(df['low'].iloc[i])
        curr=df['close'].iloc[-1]; sw=[]
        for p in ph[-5:]:
            if curr>p*1.001: sw.append({'t':'BSL','p':p})
        for p in pl[-5:]:
            if curr<p*0.999: sw.append({'t':'SSL','p':p})
        return sw

    def cvd(self,df):
        if 'volume' not in df.columns or len(df)<20: return 0
        cv=(df['volume']*np.sign(df['close'].diff().fillna(0))).cumsum()
        if df['close'].iloc[-1]<=df['close'].rolling(20).min().iloc[-1]*1.01 and cv.iloc[-1]>cv.rolling(20).min().iloc[-1]: return 15
        if df['close'].iloc[-1]>=df['close'].rolling(20).max().iloc[-1]*0.99 and cv.iloc[-1]<cv.rolling(20).max().iloc[-1]: return -10
        return 0

    def smart_sl(self,df,sig):
        av=atr(df).iloc[-1]; buf=av*0.2; cl=df['close'].iloc[-1]; o=self.obs(df)
        if sig=='LONG':
            b=df['low'].rolling(50).min().iloc[-1]
            for x in o:
                if x['t']=='bull' and b<x['p']<cl: b=x['p']
            r=b-buf; return r if np.isfinite(r) and r>0 else cl*0.98
        else:
            b=df['high'].rolling(50).max().iloc[-1]
            for x in o:
                if x['t']=='bear' and cl<x['p']<b: b=x['p']
            r=b+buf; return r if np.isfinite(r) else cl*1.02

    def kelly(self):
        wr=(sum(self.wh)/len(self.wh)) if self.wh else 0.55
        return max(0.05,min((wr*1.5-(1-wr))/1.5,0.25))*0.25

    def score(self,df,eps):
        sc=0; rs=[]; kd=self.knn(df)
        if kd!=0: sc+=20; rs.append(f"🧠 k-NN: +20 ({'rialzista' if kd==1 else 'ribassista'})")
        ev={p:df[f'EMA_{p}'].iloc[-1] for p in eps if f'EMA_{p}' in df.columns}
        sp=sorted(ev)
        if len(sp)>=2:
            if all(ev[sp[i]]>ev[sp[i+1]] for i in range(len(sp)-1)): sc+=20; rs.append("📈 EMA bullish: +20")
            elif all(ev[sp[i]]<ev[sp[i+1]] for i in range(len(sp)-1)): sc+=20; rs.append("📉 EMA bearish: +20")
        cv=self.cvd(df); sc+=cv
        if cv>0: rs.append(f"💧 CVD bull: +{cv}")
        elif cv<0: rs.append(f"⚠️ CVD bear: {cv}")
        for o in self.obs(df):
            cur=df['close'].iloc[-1]
            if o['t']=='bull' and cur<o['p']*1.01: sc+=10; rs.append("🏛️ OB Bullish: +10"); break
            elif o['t']=='bear' and cur>o['p']*0.99: sc+=10; rs.append("🏛️ OB Bearish: +10"); break
        for sw in self.sweeps(df): sc+=25; rs.append(f"🎯 {sw['t']} Sweep: +25")
        if 'MACD_line' in df.columns and df['MACD_line'].iloc[-1]>df['MACD_signal'].iloc[-1]: sc+=5; rs.append("📊 MACD: +5")
        if 'RSI' in df.columns and 40<df['RSI'].iloc[-1]<60: sc+=5; rs.append("📊 RSI neutrale: +5")
        if 'SUPERTREND_DIR' in df.columns:
            st=df['SUPERTREND_DIR'].iloc[-1]
            if st==1: sc+=10; rs.append("🌊 SuperTrend bull: +10")
            elif st==-1: sc+=10; rs.append("🌊 SuperTrend bear: +10")
        mult=self.regime(df); orig=sc; sc=int(min(sc*mult,100))
        if mult<1: rs.append(f"⚡ Regime: x{mult:.2f} ({orig}→{sc})")
        return sc, rs, kd

    def signal(self,df,eps,thr=60):
        ai,rs,kd=self.score(df,eps); mult=self.regime(df)
        if ai>=thr and mult>=0.7:
            sig="LONG" if kd==1 or ai>=75 else ("SHORT" if kd==-1 else "NEUTRAL")
        else: sig="NEUTRAL"
        sl=self.smart_sl(df,sig) if sig in("LONG","SHORT") else None
        ks=self.kelly()           if sig in("LONG","SHORT") else None
        return {"signal":sig,"confidence":ai,"reasons":rs,"smart_sl":sl,
                "kelly_size":ks,"knn_dir":kd,"regime":mult}


# ============================================================================
# 6. GRAFICO
# ============================================================================

def create_chart(df, ema_periods, show, source_label):
    has = lambda x: x in show
    rows_cfg = [("📈 Prezzo & Indicatori", 0.50),("📊 Volume", 0.12)]
    smap = {}
    for key,lbl,h in [("macd","⚡ MACD",0.14),("rsi","📉 RSI",0.12),
                       ("stoch","🔄 Stocastico",0.12),("adx","💪 ADX/DI",0.12),("cci","📐 CCI",0.10)]:
        flag={"macd":has("MACD"),"rsi":has("RSI"),"stoch":has("Stochastic"),
              "adx":has("ADX"),"cci":has("CCI")}[key]
        if flag: smap[key]=len(rows_cfg)+1; rows_cfg.append((lbl,h))

    nr=len(rows_cfg); ttl=sum(r[1] for r in rows_cfg)
    fig=make_subplots(rows=nr,cols=1,shared_xaxes=True,vertical_spacing=0.025,
        row_heights=[r[1]/ttl for r in rows_cfg],
        subplot_titles=[r[0] for r in rows_cfg])

    INC,DEC="#26a69a","#ef5350"
    fig.add_trace(go.Candlestick(x=df.index,open=df["open"],high=df["high"],
        low=df["low"],close=df["close"],name="Prezzo",
        increasing_line_color=INC,decreasing_line_color=DEC,
        increasing_fillcolor=INC,decreasing_fillcolor=DEC),row=1,col=1)

    EC={5:"#00e5ff",10:"#69f0ae",20:"#ffeb3b",50:"#ff9800",100:"#ce93d8",200:"#ef9a9a"}
    for p in ema_periods:
        cn=f"EMA_{p}"
        if cn in show and cn in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[cn],name=f"EMA {p}",
                line=dict(color=EC.get(p,"#aaa"),width=1.5)),row=1,col=1)

    if has("Bollinger Bands") and "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_upper"],name="BB Upper",
            line=dict(color="rgba(100,150,255,0.7)",width=1,dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_lower"],name="BB Lower",
            line=dict(color="rgba(100,150,255,0.7)",width=1,dash="dot"),
            fill='tonexty',fillcolor='rgba(100,150,255,0.04)'),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_middle"],name="BB Mid",
            line=dict(color="rgba(180,180,180,0.35)",width=1)),row=1,col=1)

    if has("VWAP") and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["VWAP"],name="VWAP",
            line=dict(color="#ff6b6b",width=2,dash="dashdot")),row=1,col=1)

    if has("SuperTrend") and "SUPERTREND" in df.columns:
        bull=df["SUPERTREND"].where(df["SUPERTREND_DIR"]==1)
        bear=df["SUPERTREND"].where(df["SUPERTREND_DIR"]==-1)
        fig.add_trace(go.Scatter(x=df.index,y=bull,name="ST ↑",
            line=dict(color="#00e676",width=2),connectgaps=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=bear,name="ST ↓",
            line=dict(color="#ff1744",width=2),connectgaps=False),row=1,col=1)

    vc=[INC if c>=o else DEC for c,o in zip(df["close"],df["open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["volume"],name="Volume",marker_color=vc,opacity=0.65),row=2,col=1)

    if has("MACD") and "macd" in smap:
        r=smap["macd"]; hc=[INC if v>=0 else DEC for v in df["MACD_hist"]]
        fig.add_trace(go.Bar(x=df.index,y=df["MACD_hist"],name="MACD Hist",marker_color=hc,opacity=0.7),row=r,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["MACD_line"],name="MACD",line=dict(color="#2196f3",width=1.5)),row=r,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["MACD_signal"],name="Signal",line=dict(color="#ff9800",width=1.5)),row=r,col=1)
        fig.add_hline(y=0,line_color="rgba(255,255,255,0.15)",line_width=1,row=r,col=1)

    if has("RSI") and "rsi" in smap:
        r=smap["rsi"]
        fig.add_trace(go.Scatter(x=df.index,y=df["RSI"],name="RSI",
            line=dict(color="#ce93d8",width=2),fill='tozeroy',fillcolor='rgba(206,147,216,0.05)'),row=r,col=1)
        for lv,col in [(70,"rgba(239,83,80,0.7)"),(50,"rgba(200,200,200,0.2)"),(30,"rgba(38,166,154,0.7)")]:
            fig.add_hline(y=lv,line_dash="dash",line_color=col,line_width=1,row=r,col=1)
        fig.update_yaxes(range=[0,100],row=r,col=1)

    if has("Stochastic") and "stoch" in smap:
        r=smap["stoch"]
        fig.add_trace(go.Scatter(x=df.index,y=df["STOCH_K"],name="%K",line=dict(color="#4fc3f7",width=1.5)),row=r,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["STOCH_D"],name="%D",line=dict(color="#ff8a65",width=1.5)),row=r,col=1)
        for lv,col in [(80,"rgba(239,83,80,0.7)"),(20,"rgba(38,166,154,0.7)")]:
            fig.add_hline(y=lv,line_dash="dash",line_color=col,line_width=1,row=r,col=1)
        fig.update_yaxes(range=[0,100],row=r,col=1)

    if has("ADX") and "adx" in smap:
        r=smap["adx"]
        fig.add_trace(go.Scatter(x=df.index,y=df["ADX"],name="ADX",line=dict(color="#ffeb3b",width=2)),row=r,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["DI_plus"],name="DI+",line=dict(color="#69f0ae",width=1.2,dash="dot")),row=r,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["DI_minus"],name="DI-",line=dict(color="#ef9a9a",width=1.2,dash="dot")),row=r,col=1)
        fig.add_hline(y=25,line_dash="dash",line_color="rgba(255,255,255,0.25)",line_width=1,row=r,col=1)

    if has("CCI") and "cci" in smap:
        r=smap["cci"]
        fig.add_trace(go.Scatter(x=df.index,y=df["CCI"],name="CCI",line=dict(color="#80cbc4",width=1.5)),row=r,col=1)
        for lv,col in [(100,"rgba(239,83,80,0.7)"),(0,"rgba(200,200,200,0.2)"),(-100,"rgba(38,166,154,0.7)")]:
            fig.add_hline(y=lv,line_dash="dash",line_color=col,line_width=1,row=r,col=1)

    fig.update_layout(
        title=dict(text=f"📊 Jarvis V7  |  Fonte: {source_label}",font=dict(size=16,color="#e0e0e0"),x=0.01),
        template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#161b22",
        height=max(680,480+nr*75),showlegend=True,
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,
            bgcolor="rgba(22,27,34,0.85)",bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,font=dict(size=10)),
        hovermode="x unified",margin=dict(l=55,r=15,t=75,b=35))
    for i in range(1,nr+1):
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)",zeroline=False,rangeslider_visible=False,row=i,col=1)
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)",zeroline=False,row=i,col=1)
    return fig


# ============================================================================
# 7. INTERFACCIA PRINCIPALE
# ============================================================================

def main():
    st.markdown("""<style>
    [data-testid="stMetric"]{background:#161b22;border-radius:8px;padding:10px 14px;border:1px solid #30363d;}
    </style>""", unsafe_allow_html=True)

    st.title("🧠 Jarvis V7 — Crypto Dashboard")
    st.caption("🔵 Binance (alta risoluzione)  •  🟣 CoinGecko (tutti i coin)  •  SMC • AI Score • Calendario")
    st.divider()

    if 'jarvis' not in st.session_state:
        st.session_state.jarvis = JarvisV7Engine()
    jarvis = st.session_state.jarvis

    # ── SIDEBAR ──
    with st.sidebar:
        st.header("⚙️ Configurazione")
        with st.spinner("Caricamento lista cripto (CoinGecko + Binance)..."):
            crypto_list = build_crypto_list()

        if not crypto_list:
            st.error("Impossibile caricare la lista. Riprova.")
            st.stop()

        # Ricerca testuale
        search = st.text_input("🔍 Cerca coin (nome o simbolo)", "")
        filtered_list = [c for c in crypto_list
                         if search.upper() in c["symbol"] or search.lower() in c["name"].lower()
                        ] if search else crypto_list

        sel_display = st.selectbox("Seleziona Criptovaluta",
                                   [c["display"] for c in filtered_list[:300]])
        coin_info = next((c for c in filtered_list if c["display"]==sel_display), crypto_list[0])

        src_badge = "🔵 Binance" if coin_info["on_binance"] else "🟣 CoinGecko"
        st.info(f"**Fonte dati:** {src_badge}  \n**Coin:** {coin_info['name']}  \n**Rank:** #{coin_info['rank']}")

        timeframe = st.selectbox("⏱️ Timeframe",
            ["1m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w"],index=4)

        if not coin_info["on_binance"]:
            cg_days, cg_gran = TF_TO_CG.get(timeframe,(30,"~4h"))
            st.caption(f"⚠️ CoinGecko: granularità reale ≈ **{cg_gran}** (days={cg_days})")

        limit = st.slider("📦 Candele (Binance)", 100, 500, 250, step=50)

        st.divider()
        st.subheader("📈 Indicatori")
        ema_opts = ["EMA_5","EMA_10","EMA_20","EMA_50","EMA_100","EMA_200"]
        sel_emas = st.multiselect("EMA", ema_opts, default=["EMA_20","EMA_50","EMA_200"])
        ema_periods = [int(e.split("_")[1]) for e in sel_emas] or [50]

        overlay = st.multiselect("Overlay",["Bollinger Bands","VWAP","SuperTrend"],
                                  default=["Bollinger Bands","SuperTrend"])
        subs = st.multiselect("Sottofinestre",["MACD","RSI","Stochastic","ADX","CCI"],
                               default=["MACD","RSI","ADX"])
        show = sel_emas + overlay + subs

        st.divider()
        soglia = st.slider("🎯 Confidenza min %", 40, 80, 60)
        st.divider()
        if st.button("🔄 Aggiorna", type="primary", use_container_width=True):
            st.cache_data.clear(); st.rerun()

        st.info("**🧠 Moduli:** k-NN • Regime Filter\nOrder Blocks • Liquidity Sweep\n"
                f"CVD • Smart SL • Kelly\n\n**Soglia:** {soglia}%")

    # ── AREA PRINCIPALE ──
    col_chart, col_cal = st.columns([3,1])

    with col_chart:
        with st.spinner(f"Scaricamento dati {coin_info['symbol']}..."):
            df_raw, src_used, tf_used = get_ohlcv(coin_info, timeframe, limit)

        if df_raw.empty:
            st.error(f"❌ Nessun dato disponibile per **{coin_info['name']}**.")
            st.stop()

        df = add_indicators(df_raw, ema_periods)
        res = jarvis.signal(df, ema_periods, soglia)
        if len(df)>2:
            jarvis.update(df, 1 if df['close'].iloc[-1]>df['close'].iloc[-2] else -1)

        sig=res["signal"]; sc=res["confidence"]
        knn_t="🟢 LONG" if res['knn_dir']==1 else ("🔴 SHORT" if res['knn_dir']==-1 else "⚪ NEUTRO")

        if   sig=="LONG":  st.success(f"🔵 **LONG — AI Score: {sc}/100**  |  Fonte: {src_used} ({tf_used})")
        elif sig=="SHORT": st.error(  f"🔴 **SHORT — AI Score: {sc}/100**  |  Fonte: {src_used} ({tf_used})")
        else:              st.warning(f"⚪ **NESSUN SEGNALE — AI Score: {sc}/100**  |  Fonte: {src_used} ({tf_used})")

        # KPI
        curr=df['close'].iloc[-1]; prev=df['close'].iloc[-2]
        dpct=(curr/prev-1)*100 if prev else 0
        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("💰 Prezzo",   f"${curr:,.4f}",   f"{dpct:+.2f}%")
        c2.metric("📉 RSI",      f"{df['RSI'].iloc[-1]:.1f}"   if 'RSI' in df.columns else "—")
        c3.metric("💪 ADX",      f"{df['ADX'].iloc[-1]:.1f}"   if 'ADX' in df.columns else "—")
        c4.metric("📊 ATR",      f"{df['ATR'].iloc[-1]:.4f}"   if 'ATR' in df.columns else "—")
        bbw=((df['BB_upper'].iloc[-1]-df['BB_lower'].iloc[-1])/df['BB_middle'].iloc[-1]*100) if 'BB_upper' in df.columns else 0
        c5.metric("📐 BB Width", f"{bbw:.1f}%")
        st_d=df['SUPERTREND_DIR'].iloc[-1] if 'SUPERTREND_DIR' in df.columns else 0
        c6.metric("🌊 SuperTrend","🟢 BULL" if st_d==1 else "🔴 BEAR")

        with st.expander("🧠 Dettaglio AI Score"):
            ca,cb=st.columns(2)
            with ca:
                st.write(f"**Score:** {sc}/100 | **k-NN:** {knn_t}")
                st.write(f"**Regime:** x{res['regime']:.2f}")
                if res.get('smart_sl'):
                    rsk=abs((res['smart_sl']-curr)/curr)*100
                    st.write(f"**Smart SL:** ${res['smart_sl']:.4f}  ({rsk:.2f}%)")
                if res.get('kelly_size'):
                    st.write(f"**Kelly:** {res['kelly_size']*100:.1f}% capitale")
            with cb:
                for r in res['reasons']: st.write(f"• {r}")

        st.subheader("📈 Grafico Tecnico")
        fig=create_chart(df, ema_periods, show, f"{src_used} | {coin_info['name']} | {tf_used}")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Dati recenti"):
            cols_d=["open","high","low","close","volume","RSI","MACD_line","ADX","STOCH_K","ATR","VWAP","CCI"]
            st.dataframe(df.tail(20)[[c for c in cols_d if c in df.columns]].round(4),use_container_width=True)

    # ── CALENDARIO ──
    with col_cal:
        st.subheader("📅 Calendario Economico")
        with st.spinner("Caricamento..."):
            all_ev=get_economic_calendar()
        c1d,c2d=st.columns(2)
        with c1d: sd=st.date_input("Da",datetime.now().date())
        with c2d: ed=st.date_input("A", datetime.now().date()+timedelta(days=14))
        imp=st.multiselect("Impatto",["HIGH","MEDIUM","LOW"],default=["HIGH","MEDIUM"])
        evs=sorted([e for e in all_ev if e.get("date") and imp
                    and sd<=datetime.strptime(e["date"],"%Y-%m-%d").date()<=ed
                    and e["impact"] in imp],
                   key=lambda x:x["date"]+x.get("time",""))
        if evs:
            for ev in evs:
                icon={"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(ev["impact"],"⚪")
                act=f" → **{ev['actual']}**" if ev.get("actual") else ""
                st.markdown(f"{icon} **{ev['event']}**{act}")
                st.caption(f"📍{ev['country']} | {ev['date']} {ev.get('time','')}")
                st.caption(f"Prec:`{ev['prev']}` Prev:`{ev['forecast']}`")
                st.divider()
        else:
            st.info("Nessun evento nel periodo")
        st.caption("💡 HIGH = alta volatilità attesa")
        st.divider()
        st.subheader("🏆 Top 5 Market Cap")
        for c in crypto_list[:5]:
            col="limegreen" if c.get('change',0)>=0 else "tomato"
            ar="▲" if c.get('change',0)>=0 else "▼"
            src_i="🔵" if c["on_binance"] else "🟣"
            st.markdown(
                f"{src_i} **{c['symbol']}** ${c['price']:,.4f} "
                f"<span style='color:{col}'>{ar}{c.get('change',0):+.2f}%</span><br>"
                f"<small style='color:#666'>{c['name']} · Vol ${c['volume']/1e9:.1f}B</small>",
                unsafe_allow_html=True)

if __name__=="__main__":
    main()
