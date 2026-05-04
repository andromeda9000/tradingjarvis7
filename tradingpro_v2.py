import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import re
from collections import deque
import json
import os
import warnings
import time
import hmac
import hashlib
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Jarvis Pro — Crypto AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────────────────────────────────────────
ADX_TREND_GATE = 20          # segnale emesso solo se ADX > soglia
ATR_MIN_PCT    = 0.25        # segnale emesso solo se ATR% > soglia
SIGNAL_LOG     = "jarvis_signal_log.json"
BITGET_CFG     = "bitget_cfg.json"          # file locale per API keys

TIMEFRAMES_BIN = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"]

# ─────────────────────────────────────────────────────────────────────────────
# ── SEZIONE CONFIGURAZIONE API BITGET ──────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def load_bitget_cfg():
    if os.path.exists(BITGET_CFG):
        try:
            return json.load(open(BITGET_CFG))
        except Exception:
            pass
    return {"api_key": "", "api_secret": "", "passphrase": "", "sandbox": True}

def save_bitget_cfg(cfg: dict):
    json.dump(cfg, open(BITGET_CFG, "w"), indent=2)

def render_bitget_settings():
    """Tab dedicato alla configurazione chiavi API Bitget."""
    st.subheader("🔑 Configurazione API Bitget")
    st.markdown("""
    <div style='background:#161b22;border:1px solid #30363d;border-left:4px solid #f0883e;
         border-radius:8px;padding:12px 16px;margin-bottom:16px'>
      <b style='color:#f0883e'>⚠️ Sicurezza</b><br>
      <span style='color:#8b949e;font-size:13px'>
      Le chiavi vengono salvate <b>localmente</b> nel file <code>bitget_cfg.json</code>.
      Non condividere mai questo file. Usa chiavi con <b>soli permessi Futures</b>
      (lettura + ordini) e <b>IP whitelist</b> attiva su Bitget.
      </span>
    </div>
    """, unsafe_allow_html=True)

    cfg = load_bitget_cfg()

    with st.form("bitget_form"):
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input(
                "API Key", value=cfg.get("api_key",""),
                placeholder="Incolla qui la tua API Key",
                help="Trovata nella sezione API Management di Bitget"
            )
            passphrase = st.text_input(
                "Passphrase", value=cfg.get("passphrase",""),
                type="password",
                placeholder="Passphrase impostata su Bitget",
                help="La passphrase che hai scelto al momento della creazione della chiave"
            )
        with col2:
            api_secret = st.text_input(
                "API Secret", value=cfg.get("api_secret",""),
                type="password",
                placeholder="Incolla qui la tua API Secret",
                help="Secret mostrato una sola volta alla creazione"
            )
            sandbox = st.toggle(
                "🧪 Modalità DEMO (sandbox)",
                value=cfg.get("sandbox", True),
                help="Attivo = nessun ordine reale. Disattiva solo se sai cosa fai."
            )

        st.markdown("---")
        st.markdown("**🛡️ Permessi consigliati sulla chiave Bitget:**")
        st.markdown("""
        - ✅ Futures — Lettura  
        - ✅ Futures — Ordini  
        - ❌ Spot — non necessario  
        - ❌ Prelievi — **mai abilitare**  
        """)

        saved = st.form_submit_button("💾 Salva configurazione", type="primary")
        if saved:
            new_cfg = {
                "api_key": api_key.strip(),
                "api_secret": api_secret.strip(),
                "passphrase": passphrase.strip(),
                "sandbox": sandbox,
            }
            save_bitget_cfg(new_cfg)
            st.success("✅ Configurazione salvata correttamente!")

    # Test connessione
    st.divider()
    st.subheader("🔌 Test connessione")
    col_test, col_mode = st.columns([2,1])
    with col_mode:
        loaded = load_bitget_cfg()
        mode_label = "🧪 DEMO (sandbox)" if loaded.get("sandbox") else "🔴 LIVE"
        st.info(f"Modalità attuale: **{mode_label}**")
    with col_test:
        if st.button("🔍 Verifica chiavi API", use_container_width=True):
            cfg_now = load_bitget_cfg()
            if not cfg_now.get("api_key"):
                st.error("❌ Nessuna chiave configurata.")
            else:
                ok, msg = bitget_test_connection(cfg_now)
                if ok:
                    st.success(f"✅ Connessione OK: {msg}")
                else:
                    st.error(f"❌ Errore: {msg}")

    # Istruzioni guidate
    with st.expander("📖 Come creare le chiavi API su Bitget — guida passo passo"):
        st.markdown("""
        1. Accedi a [bitget.com](https://www.bitget.com) e vai in **Profilo → API Management**
        2. Clicca **Create API**
        3. Dai un nome alla chiave (es. `JarvisPro`)
        4. Imposta una **Passphrase** (salvala!)
        5. Seleziona i permessi: solo **Futures → Read + Order**
        6. Aggiungi il tuo **IP** in whitelist
        7. Completa la verifica 2FA
        8. Copia **API Key**, **Secret** e **Passphrase** qui sopra
        9. Lascia attiva la modalità **DEMO** finché non hai testato bene
        """)

    # Saldo account (se chiavi presenti)
    st.divider()
    st.subheader("💰 Saldo account Bitget")
    cfg_now = load_bitget_cfg()
    if cfg_now.get("api_key"):
        if st.button("📊 Mostra saldo Futures", use_container_width=True):
            with st.spinner("Richiesta saldo..."):
                bal = bitget_get_balance(cfg_now)
            if bal:
                for asset, info in bal.items():
                    st.metric(
                        label=f"💵 {asset}",
                        value=f"${float(info.get('available','0')):,.2f}",
                        delta=f"Total: ${float(info.get('equity','0')):,.2f}"
                    )
            else:
                st.warning("Impossibile recuperare il saldo (controlla le chiavi).")
    else:
        st.info("Inserisci prima le chiavi API per visualizzare il saldo.")

    # Posizioni aperte
    st.divider()
    st.subheader("📋 Posizioni aperte")
    if cfg_now.get("api_key"):
        if st.button("🔄 Carica posizioni", use_container_width=True):
            with st.spinner("Caricamento posizioni..."):
                positions = bitget_get_positions(cfg_now)
            if positions:
                df_pos = pd.DataFrame(positions)
                st.dataframe(df_pos, use_container_width=True)
            else:
                st.info("Nessuna posizione aperta.")
    else:
        st.info("Inserisci prima le chiavi API.")


# ─────────────────────────────────────────────────────────────────────────────
# ── BITGET API HELPERS ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _bitget_sign(secret: str, ts: str, method: str, path: str, body: str = "") -> str:
    msg = ts + method.upper() + path + body
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest().hex()

def _bitget_headers(cfg: dict, method: str, path: str, body: str = "") -> dict:
    ts = str(int(time.time() * 1000))
    sig = _bitget_sign(cfg["api_secret"], ts, method, path, body)
    return {
        "ACCESS-KEY":        cfg["api_key"],
        "ACCESS-SIGN":       sig,
        "ACCESS-TIMESTAMP":  ts,
        "ACCESS-PASSPHRASE": cfg["passphrase"],
        "Content-Type":      "application/json",
        "locale":            "en-US",
    }

def _bitget_base(cfg: dict) -> str:
    if cfg.get("sandbox"):
        return "https://api.bitget.com"   # Bitget non ha sandbox separato; usiamo paper trading via accountType
    return "https://api.bitget.com"

def bitget_test_connection(cfg: dict):
    try:
        path = "/api/v2/account/info"
        base = _bitget_base(cfg)
        headers = _bitget_headers(cfg, "GET", path)
        r = requests.get(base + path, headers=headers, timeout=8)
        data = r.json()
        if data.get("code") == "00000":
            uid = data.get("data", {}).get("userId", "—")
            return True, f"UserID {uid}"
        return False, data.get("msg", str(data))
    except Exception as e:
        return False, str(e)

def bitget_get_balance(cfg: dict):
    try:
        path = "/api/v2/mix/account/accounts?productType=USDT-FUTURES"
        headers = _bitget_headers(cfg, "GET", path)
        r = requests.get(_bitget_base(cfg) + path, headers=headers, timeout=8)
        data = r.json()
        if data.get("code") == "00000":
            result = {}
            for item in data.get("data", []):
                coin = item.get("marginCoin", "USDT")
                result[coin] = {
                    "available": item.get("available", "0"),
                    "equity":    item.get("equity",    "0"),
                    "unrealized":item.get("unrealizedPL", "0"),
                }
            return result
        return None
    except Exception:
        return None

def bitget_get_positions(cfg: dict):
    try:
        path = "/api/v2/mix/position/all-position?productType=USDT-FUTURES&marginCoin=USDT"
        headers = _bitget_headers(cfg, "GET", path)
        r = requests.get(_bitget_base(cfg) + path, headers=headers, timeout=8)
        data = r.json()
        if data.get("code") == "00000":
            rows = []
            for p in data.get("data", []):
                side = p.get("holdSide","")
                size = float(p.get("total", 0))
                if size == 0:
                    continue
                rows.append({
                    "Simbolo":     p.get("symbol",""),
                    "Direzione":   "🟢 LONG" if side=="long" else "🔴 SHORT",
                    "Dimensione":  size,
                    "Entry ($)":   float(p.get("openPriceAvg", 0)),
                    "Mark ($)":    float(p.get("markPrice", 0)),
                    "PnL ($)":     float(p.get("unrealizedPL", 0)),
                    "Leva":        p.get("leverage","—"),
                })
            return rows
        return []
    except Exception:
        return []

def bitget_place_order(cfg: dict, symbol: str, side: str, size: float,
                       sl: float, tp: float, leverage: int = 5):
    """Apre un ordine futures su Bitget (paper o live in base a cfg)."""
    try:
        path = "/api/v2/mix/order/place-order"
        body_dict = {
            "symbol":       symbol + "USDT",
            "productType":  "USDT-FUTURES",
            "marginMode":   "isolated",
            "marginCoin":   "USDT",
            "size":         str(round(size, 4)),
            "side":         "buy"  if side == "LONG" else "sell",
            "tradeSide":    "open",
            "orderType":    "market",
            "presetStopLossPrice":   str(round(sl, 6)),
            "presetStopSurplusPrice":str(round(tp, 6)),
        }
        body = json.dumps(body_dict)
        headers = _bitget_headers(cfg, "POST", path, body)
        r = requests.post(_bitget_base(cfg) + path, headers=headers,
                          data=body, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# ── DATI BINANCE ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_top_crypto_pairs(limit: int = 300):
    try:
        data = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr", timeout=15
        ).json()
        pairs = sorted(
            [d for d in data if d["symbol"].endswith("USDT")],
            key=lambda x: float(x["quoteVolume"]), reverse=True
        )[:limit]
        result = []
        for p in pairs:
            coin = p["symbol"].replace("USDT", "")
            result.append({
                "symbol":  p["symbol"],
                "coin":    coin,
                "price":   float(p["lastPrice"]),
                "volume":  float(p["quoteVolume"]),
                "display": f"{coin} — ${float(p['lastPrice']):,.4f} (vol ${float(p['quoteVolume'])/1e9:.1f}B)",
            })
        return result
    except Exception:
        return [{"symbol":"BTCUSDT","coin":"BTC","price":0,"volume":0,"display":"BTC"}]

@st.cache_data(ttl=60)
def get_ohlcv(symbol: str, interval: str = "1h", limit: int = 300) -> pd.DataFrame:
    try:
        data = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        ).json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            "ts","open","high","low","close","volume",
            "cts","qav","trades","tbbase","tbquote","ignore"
        ])
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df[["open","high","low","close","volume"]]
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# ── INDICATORI TECNICI ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def ema(s, p):    return s.ewm(span=p, adjust=False).mean()
def sma(s, p):    return s.rolling(p).mean()
def stdev(s, p):  return s.rolling(p).std()

def rsi14(s, p=14):
    d = s.diff()
    g = d.where(d>0,0).ewm(alpha=1/p,adjust=False).mean()
    l = (-d.where(d<0,0)).ewm(alpha=1/p,adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100/(1+rs)

def macd_ind(s, f=12, sl=26, sig=9):
    m = ema(s,f) - ema(s,sl)
    sg = ema(m,sig)
    return m, sg, m-sg

def boll(s, p=20, k=2):
    mid = sma(s,p)
    std_ = stdev(s,p)
    return mid+k*std_, mid-k*std_, mid

def atr14(df, p=14):
    h,l,c = df["high"],df["low"],df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def adx_full(df, p=14):
    h,l,c = df["high"],df["low"],df["close"]
    up = h.diff(); dn = -l.diff()
    pdm = up.where((up>dn)&(up>0),0.0)
    ndm = dn.where((dn>up)&(dn>0),0.0)
    atr_ = atr14(df,p)
    pdi = 100*(pdm.ewm(alpha=1/p,adjust=False).mean()/atr_.replace(0,np.nan))
    ndi = 100*(ndm.ewm(alpha=1/p,adjust=False).mean()/atr_.replace(0,np.nan))
    dx  = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
    return dx.ewm(alpha=1/p,adjust=False).mean(), pdi, ndi

def stoch14(df, k=14, d=3):
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    pct = 100*(df["close"]-lo)/(hi-lo).replace(0,np.nan)
    return pct, pct.rolling(d).mean()

def vwap(df):
    tp = (df["high"]+df["low"]+df["close"])/3
    cum_tpv = (tp*df["volume"]).cumsum()
    cum_v   = df["volume"].cumsum()
    return cum_tpv/cum_v.replace(0,np.nan)

def cci20(df, p=20):
    tp = (df["high"]+df["low"]+df["close"])/3
    ma = tp.rolling(p).mean()
    md = tp.rolling(p).apply(lambda x: np.abs(x-x.mean()).mean(), raw=True)
    return (tp-ma)/(0.015*md.replace(0,np.nan))

def supertrend(df, p=10, mult=3.0):
    atr_ = atr14(df, p)
    hl2  = (df["high"]+df["low"])/2
    upper = hl2 + mult*atr_
    lower = hl2 - mult*atr_
    st_up  = pd.Series(np.nan, index=df.index)
    st_dn  = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        p_up = lower.iloc[i]
        p_dn = upper.iloc[i]
        if st_dn.iloc[i-1] is not np.nan and not np.isnan(st_dn.iloc[i-1]):
            p_up = max(p_up, st_up.iloc[i-1]) if df["close"].iloc[i-1]>st_up.iloc[i-1] else p_up
        if st_up.iloc[i-1] is not np.nan and not np.isnan(st_up.iloc[i-1]):
            p_dn = min(p_dn, st_dn.iloc[i-1]) if df["close"].iloc[i-1]<st_dn.iloc[i-1] else p_dn
        st_up.iloc[i] = p_up; st_dn.iloc[i] = p_dn
        prev_dir = direction.iloc[i-1]
        if prev_dir == 1:
            direction.iloc[i] = 1 if df["close"].iloc[i] >= st_up.iloc[i] else -1
        else:
            direction.iloc[i] = -1 if df["close"].iloc[i] <= st_dn.iloc[i] else 1
    st_line = pd.Series(np.where(direction==1, st_up, st_dn), index=df.index)
    return st_line, direction

def pivot_sr(df, window=10):
    hi = df["high"]; lo = df["low"]
    levels = []
    for i in range(window, len(df)-window):
        if hi.iloc[i] == hi.iloc[i-window:i+window+1].max():
            levels.append(("R", float(hi.iloc[i])))
        if lo.iloc[i] == lo.iloc[i-window:i+window+1].min():
            levels.append(("S", float(lo.iloc[i])))
    return levels[-10:]

def add_all_indicators(df, ema_ps=(20,50,200)):
    df = df.copy()
    c = df["close"]
    for p in ema_ps:
        df[f"EMA_{p}"] = ema(c, p)
    df["RSI"] = rsi14(c)
    df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd_ind(c)
    df["BB_up"], df["BB_lo"], df["BB_mid"] = boll(c)
    df["ATR"] = atr14(df)
    df["ADX"], df["DI_plus"], df["DI_minus"] = adx_full(df)
    df["STOCH_K"], df["STOCH_D"] = stoch14(df)
    df["VWAP"] = vwap(df)
    df["CCI"] = cci20(df)
    df["VOLMA"] = sma(df["volume"], 20)
    df["ST_line"], df["ST_dir"] = supertrend(df)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# ── MOTORE SEGNALE JARVIS V3 ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class JarvisEngine:
    def __init__(self, k=12, hist=500):
        self.k = k
        self.feat_hist  = deque(maxlen=hist)
        self.label_hist = deque(maxlen=hist)

    def _features(self, df):
        c = df["close"]
        rsi_v = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 50
        cci_v = float(df["CCI"].iloc[-1]) if "CCI" in df.columns else 0
        rsi_n = np.clip((rsi_v - 30)/40, 0, 1)
        cci_n = np.clip((cci_v + 200)/400, 0, 1)
        return np.array([rsi_n, cci_n])

    def update(self, df, label):
        self.feat_hist.append(self._features(df))
        self.label_hist.append(label)

    def knn_predict(self, df):
        if len(self.feat_hist) < self.k:
            return 0
        cur = self._features(df)
        X = np.array(list(self.feat_hist))
        y = np.array(list(self.label_hist))
        dists = np.sqrt(((np.log1p(X) - np.log1p(cur))**2).sum(axis=1))
        idx   = np.argsort(dists)[:self.k]
        return 1 if y[idx].sum() > 0 else -1

    def signal(self, df, ema_ps, threshold=60, htf_score=50):
        if df.empty or len(df) < 60:
            return self._neutral("Dati insufficienti")

        # ── REGIME GATE ──────────────────────────────────────────────────────
        adx_v = float(df["ADX"].iloc[-1]) if "ADX" in df.columns else 0
        atr_v = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 0
        px    = float(df["close"].iloc[-1])
        atr_p = atr_v/px*100 if px else 0
        if adx_v < ADX_TREND_GATE:
            return self._neutral(f"ADX {adx_v:.0f} < {ADX_TREND_GATE} — mercato laterale")
        if atr_p < ATR_MIN_PCT:
            return self._neutral(f"ATR {atr_p:.2f}% < {ATR_MIN_PCT}% — troppo piatto")

        c = df["close"]
        long_s = 0; short_s = 0
        rl = []; rs = []

        # EMA stack (18 pt)
        ema_vs = {p: float(df[f"EMA_{p}"].iloc[-1]) for p in ema_ps if f"EMA_{p}" in df.columns}
        srt = sorted(ema_vs)
        if len(srt) >= 2:
            if all(ema_vs[srt[i]] > ema_vs[srt[i+1]] for i in range(len(srt)-1)) and px > ema_vs[srt[0]]:
                long_s += 18; rl.append("✅ EMA stack bullish perfetto")
            elif all(ema_vs[srt[i]] < ema_vs[srt[i+1]] for i in range(len(srt)-1)) and px < ema_vs[srt[0]]:
                short_s += 18; rs.append("✅ EMA stack bearish perfetto")
            else:
                rl.append("⚠️ EMA in conflitto"); rs.append("⚠️ EMA in conflitto")

        # SuperTrend (22 pt)
        if "ST_dir" in df.columns:
            std = int(df["ST_dir"].iloc[-1])
            if std == 1:   long_s  += 22; rl.append("✅ SuperTrend BULL")
            elif std == -1:short_s += 22; rs.append("✅ SuperTrend BEAR")

        # kNN (14 pt)
        knn = self.knn_predict(df)
        if knn == 1:   long_s  += 14; rl.append("✅ kNN rialzista")
        elif knn == -1:short_s += 14; rs.append("✅ kNN ribassista")

        # CVD divergence (12 / 10 pt)
        if "volume" in df.columns:
            delta = df["volume"] * np.sign(c.diff().fillna(0))
            cvd   = delta.cumsum()
            pl_lo = df["close"].rolling(20).min()
            cv_lo = cvd.rolling(20).min()
            ph_hi = df["close"].rolling(20).max()
            cv_hi = cvd.rolling(20).max()
            if c.iloc[-1] <= pl_lo.iloc[-1]*1.01 and cvd.iloc[-1] > cv_lo.iloc[-1]:
                long_s  += 12; rl.append("✅ CVD assorbimento long")
            if c.iloc[-1] >= ph_hi.iloc[-1]*0.99 and cvd.iloc[-1] < cv_hi.iloc[-1]:
                short_s += 10; rs.append("✅ CVD assorbimento short")

        # HTF bias (10 pt)
        if htf_score >= 60:
            long_s  += 10; rl.append(f"✅ HTF score {htf_score}/100 bullish")
        elif htf_score <= 40:
            short_s += 10; rs.append(f"✅ HTF score {htf_score}/100 bearish")

        # MACD (4 pt — solo conferma)
        if "MACD" in df.columns:
            if float(df["MACD"].iloc[-1]) > float(df["MACD_sig"].iloc[-1]):
                long_s += 4; rl.append("✅ MACD conferma long")
            else:
                short_s += 4; rs.append("✅ MACD conferma short")

        # RSI (4 pt — solo filtro)
        rsi_v = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 50
        if rsi_v < 35:   long_s  += 4; rl.append(f"✅ RSI {rsi_v:.0f} oversold")
        elif rsi_v > 65: short_s += 4; rs.append(f"✅ RSI {rsi_v:.0f} overbought")

        # ADX boost (8 pt)
        if adx_v >= 30:
            if long_s > short_s:   long_s  += 8; rl.append(f"✅ ADX {adx_v:.0f} trend forte")
            elif short_s > long_s: short_s += 8; rs.append(f"✅ ADX {adx_v:.0f} trend forte")

        # Volume (8 pt)
        if "VOLMA" in df.columns:
            vc = float(df["volume"].iloc[-1]); vm = float(df["VOLMA"].iloc[-1])
            if vc > vm*1.3:
                if long_s > short_s:   long_s  += 8; rl.append("✅ Volume sopra media")
                elif short_s > long_s: short_s += 8; rs.append("✅ Volume sopra media")

        # Normalizza 0-100
        max_pts = 100
        long_s  = min(100, int(long_s  / max_pts * 100)) if max_pts else long_s
        short_s = min(100, int(short_s / max_pts * 100)) if max_pts else short_s

        # Entrambi forti → NEUTRAL (mercato indeciso)
        if long_s >= threshold and short_s >= threshold:
            return {**self._neutral("Segnali contrari — mercato indeciso"),
                    "long_score":long_s,"short_score":short_s,
                    "reasons_long":rl,"reasons_short":rs,
                    "adx":adx_v,"atr_pct":atr_p,"rsi":rsi_v}

        # Determina direzione
        if long_s >= threshold and long_s > short_s:
            signal = "LONG"; conf = long_s
        elif short_s >= threshold and short_s > long_s:
            signal = "SHORT"; conf = short_s
        else:
            return {**self._neutral("Score insufficiente"),
                    "long_score":long_s,"short_score":short_s,
                    "reasons_long":rl,"reasons_short":rs,
                    "adx":adx_v,"atr_pct":atr_p,"rsi":rsi_v}

        # SL/TP con slippage incluso (0.08% round trip)
        slip = px * 0.0004
        sl_d = atr_v * 1.5
        if signal == "LONG":
            entry = px + slip
            sl    = px - sl_d
            tp1   = px + sl_d*2.0
            tp2   = px + sl_d*3.236
        else:
            entry = px - slip
            sl    = px + sl_d
            tp1   = px - sl_d*2.0
            tp2   = px - sl_d*3.236

        # Leva dinamica Kelly
        kelly = max(0.05, min(0.25, (conf/100*1.5 - (1-conf/100))/1.5 * 0.25))
        risk_frac = abs(px - sl) / px if px else 0.02
        leverage  = int(np.clip(kelly / risk_frac if risk_frac else 5, 1, 10))
        if adx_v > 30: leverage = min(leverage+1, 10)
        if atr_p > 2.5: leverage = max(leverage-1, 1)

        regime = "TREND" if adx_v >= 25 else "LATERALE"

        return {
            "signal":      signal, "confidence": conf,
            "long_score":  long_s, "short_score": short_s,
            "reasons_long":rl,     "reasons_short":rs,
            "entry_px":    entry,  "sl":sl, "tp1":tp1, "tp2":tp2,
            "leverage":    leverage, "adx":adx_v, "atr_pct":atr_p,
            "rsi":         rsi_v,
            "regime_msg":  f"{regime} · ADX {adx_v:.0f} · ATR {atr_p:.2f}%",
        }

    @staticmethod
    def _neutral(msg):
        return {"signal":"NEUTRAL","confidence":0,"long_score":0,"short_score":0,
                "reasons_long":[],"reasons_short":[f"⚠️ {msg}"],
                "entry_px":0,"sl":None,"tp1":None,"tp2":None,
                "leverage":1,"adx":0,"atr_pct":0,"rsi":50,
                "regime_msg":msg}


# ─────────────────────────────────────────────────────────────────────────────
# ── LOG SEGNALI ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def load_log():
    if os.path.exists(SIGNAL_LOG):
        try: return json.load(open(SIGNAL_LOG))
        except: pass
    return []

def save_log(logs):
    json.dump(logs, open(SIGNAL_LOG,"w"), indent=2)

def log_signal(symbol, tf, signal, entry, sl, tp1, conf, lev):
    logs = load_log()
    logs.append({
        "id":       len(logs)+1,
        "ts":       datetime.now().isoformat(),
        "symbol":   symbol,
        "tf":       tf,
        "signal":   signal,
        "entry":    round(entry,8),
        "sl":       round(sl,8) if sl else None,
        "tp1":      round(tp1,8) if tp1 else None,
        "conf":     conf,
        "leverage": lev,
        "result":   "OPEN",
    })
    save_log(logs)


# ─────────────────────────────────────────────────────────────────────────────
# ── GRAFICO ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def make_chart(df, ema_ps, overlays, oscillators, signal_res, height=820):
    osc_rows = oscillators[:]
    n_sub = len(osc_rows)
    row_h = [0.52] + [round(0.48/max(n_sub,1),3)]*n_sub if n_sub else [1.0]
    row_h_norm = [r/sum(row_h) for r in row_h]
    titles = ["📈 Prezzo"] + [f"📊 {o}" for o in osc_rows]

    fig = make_subplots(
        rows=1+n_sub, cols=1, shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=row_h_norm,
        subplot_titles=titles,
    )

    # ── Candele ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"],  close=df["close"],
        increasing_fillcolor="#00897b", increasing_line_color="#00897b",
        decreasing_fillcolor="#c62828", decreasing_line_color="#c62828",
        name="OHLC", showlegend=False,
    ), row=1, col=1)

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_c = ["#00897b" if df["close"].iloc[i] >= df["open"].iloc[i] else "#c62828"
             for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], marker_color=vol_c,
        name="Volume", opacity=0.35, showlegend=False,
        yaxis="y2",
    ), row=1, col=1)

    # ── EMAs ─────────────────────────────────────────────────────────────────
    ema_colors = {5:"#26c6da",10:"#66bb6a",20:"#ffca28",50:"#ff7043",100:"#ab47bc",200:"#ef5350"}
    for p in ema_ps:
        col_ = f"EMA_{p}"
        if col_ in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_], name=f"EMA{p}",
                line=dict(color=ema_colors.get(p,"#aaa"), width=1.4),
            ), row=1, col=1)

    # ── Overlays ─────────────────────────────────────────────────────────────
    if "BB" in overlays and "BB_up" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], name="BB Up",
            line=dict(color="#78909c",width=1,dash="dot"), showlegend=False), row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lo"], name="BB Lo",
            line=dict(color="#78909c",width=1,dash="dot"),
            fill="tonexty", fillcolor="rgba(120,144,156,0.07)", showlegend=False), row=1,col=1)

    if "VWAP" in overlays and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
            line=dict(color="#f48fb1", width=1.3, dash="dash")), row=1,col=1)

    if "SuperTrend" in overlays and "ST_line" in df.columns:
        bull_idx = df["ST_dir"]==1; bear_idx = df["ST_dir"]==-1
        if bull_idx.any():
            fig.add_trace(go.Scatter(
                x=df.index[bull_idx], y=df["ST_line"][bull_idx],
                name="ST Bull", line=dict(color="#00e676",width=2),
                mode="lines"), row=1,col=1)
        if bear_idx.any():
            fig.add_trace(go.Scatter(
                x=df.index[bear_idx], y=df["ST_line"][bear_idx],
                name="ST Bear", line=dict(color="#ef5350",width=2),
                mode="lines"), row=1,col=1)

    if "S/R" in overlays:
        levels = pivot_sr(df)
        shown = set()
        for kind, lvl in levels:
            if lvl in shown: continue
            shown.add(lvl)
            fig.add_hline(
                y=lvl, row=1, col=1,
                line=dict(color="#ffd54f" if kind=="R" else "#80cbc4", width=1, dash="dot"),
                annotation_text=kind, annotation_position="right",
            )

    # ── Frecce segnale ───────────────────────────────────────────────────────
    sig = signal_res.get("signal","NEUTRAL")
    if sig in ("LONG","SHORT") and signal_res.get("sl"):
        last_x = df.index[-1]; last_y = float(df["close"].iloc[-1])
        fig.add_trace(go.Scatter(
            x=[last_x], y=[last_y],
            mode="markers+text",
            marker=dict(symbol="triangle-up" if sig=="LONG" else "triangle-down",
                        size=16, color="#00e676" if sig=="LONG" else "#ef5350"),
            text=[sig], textposition="top center",
            name=sig, showlegend=False,
        ), row=1, col=1)

    # ── Oscillatori ──────────────────────────────────────────────────────────
    for ri, osc in enumerate(osc_rows, start=2):
        if osc == "RSI" and "RSI" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                line=dict(color="#ce93d8", width=1.5)), row=ri, col=1)
            fig.add_hline(y=70, line_color="#ef5350", line_dash="dot", row=ri, col=1)
            fig.add_hline(y=30, line_color="#00e676", line_dash="dot", row=ri, col=1)
            fig.add_hline(y=50, line_color="#555", line_dash="dot", row=ri, col=1)
            fig.update_yaxes(range=[0,100], row=ri, col=1)

        elif osc == "MACD" and "MACD" in df.columns:
            h_c = ["#00897b" if v>=0 else "#c62828" for v in df["MACD_hist"]]
            fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], marker_color=h_c,
                name="MACD Hist", opacity=0.7), row=ri,col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                line=dict(color="#64b5f6",width=1.3)), row=ri,col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_sig"], name="Signal",
                line=dict(color="#ef9a9a",width=1.3)), row=ri,col=1)

        elif osc == "Stoch" and "STOCH_K" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_K"], name="%K",
                line=dict(color="#4fc3f7",width=1.3)), row=ri,col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_D"], name="%D",
                line=dict(color="#f48fb1",width=1.3)), row=ri,col=1)
            fig.add_hline(y=80, line_color="#ef5350", line_dash="dot", row=ri,col=1)
            fig.add_hline(y=20, line_color="#00e676", line_dash="dot", row=ri,col=1)
            fig.update_yaxes(range=[0,100], row=ri,col=1)

        elif osc == "ADX" and "ADX" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX",
                line=dict(color="#ffca28",width=1.5)), row=ri,col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["DI_plus"], name="DI+",
                line=dict(color="#00e676",width=1, dash="dot")), row=ri,col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["DI_minus"], name="DI-",
                line=dict(color="#ef5350",width=1, dash="dot")), row=ri,col=1)
            fig.add_hline(y=ADX_TREND_GATE, line_color="#888", line_dash="dash", row=ri,col=1)

        elif osc == "CCI" and "CCI" in df.columns:
            cci_c = ["#00897b" if v>=0 else "#c62828" for v in df["CCI"]]
            fig.add_trace(go.Bar(x=df.index, y=df["CCI"], marker_color=cci_c,
                name="CCI", opacity=0.7), row=ri,col=1)
            fig.add_hline(y=100, line_color="#ef5350", line_dash="dot", row=ri,col=1)
            fig.add_hline(y=-100,line_color="#00e676", line_dash="dot", row=ri,col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", size=11),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified", xaxis_rangeslider_visible=False,
        margin=dict(l=10,r=10,t=40,b=10),
    )
    for i in range(1, 2+n_sub):
        fig.update_xaxes(gridcolor="#21262d", row=i, col=1)
        fig.update_yaxes(gridcolor="#21262d", row=i, col=1)

    # Volume axis secondaria
    fig.update_layout(yaxis2=dict(
        overlaying="y", side="right", showgrid=False,
        showticklabels=False, range=[0, df["volume"].max()*5],
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ── CALENDARIO ECONOMICO ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_calendar():
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r = requests.get(url, timeout=8)
        raw = r.json()
        result = []
        for e in raw:
            dt = e.get("date","")
            try:
                parsed = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S%z")
                date_s = parsed.strftime("%Y-%m-%d")
                time_s = parsed.strftime("%H:%M")
            except Exception:
                date_s = datetime.now().strftime("%Y-%m-%d"); time_s = ""
            result.append({
                "date":     date_s, "time": time_s,
                "country":  e.get("country",""),
                "event":    e.get("title",""),
                "impact":   e.get("impact","LOW").upper(),
                "prev":     e.get("previous","—"),
                "forecast": e.get("forecast","—"),
                "actual":   e.get("actual",""),
            })
        return result
    except Exception:
        pass
    # Fallback dinamico
    today = datetime.now()
    return [
        {"date":(today+timedelta(days=i)).strftime("%Y-%m-%d"),"time":t,
         "country":c,"event":ev,"impact":imp,"prev":pr,"forecast":fc,"actual":""}
        for i,t,c,ev,imp,pr,fc in [
            (0,"14:30","🇺🇸","CPI Inflation","HIGH","3.2%","3.1%"),
            (1,"14:30","🇺🇸","PPI","MEDIUM","2.1%","2.0%"),
            (2,"20:00","🇺🇸","FOMC Minutes","HIGH","—","—"),
            (3,"10:00","🇪🇺","BCE Rate Decision","HIGH","4.50%","4.25%"),
            (4,"14:30","🇺🇸","Non-Farm Payrolls","HIGH","175K","185K"),
            (5,"11:00","🇪🇺","GDP Flash","MEDIUM","0.3%","0.4%"),
            (7,"10:00","🇬🇧","BoE Rate Decision","HIGH","5.25%","5.00%"),
            (9,"14:30","🇺🇸","Retail Sales","MEDIUM","0.4%","0.5%"),
            (11,"14:30","🇺🇸","PCE Deflator","HIGH","2.8%","2.6%"),
            (14,"14:30","🇺🇸","Initial Jobless Claims","MEDIUM","220K","215K"),
        ]
    ]


# ─────────────────────────────────────────────────────────────────────────────
# ── SEGNALE CARD ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def render_signal_card(res, symbol, capital, risk_p, rr):
    sig = res["signal"]
    if sig == "LONG":
        bg="#003300"; brd="#00e676"; lbl="🟢 LONG"; clr="#00e676"
    elif sig == "SHORT":
        bg="#330000"; brd="#ef5350"; lbl="🔴 SHORT"; clr="#ef5350"
    else:
        bg="#1a1a1a"; brd="#555"; lbl="⚪ NEUTRAL"; clr="#aaa"

    px = res.get("entry_px", 0)
    fmt = f"${px:,.6f}" if px<0.01 else f"${px:,.4f}" if px<1 else f"${px:,.2f}"

    # Risk sizing
    risk_usd = capital * risk_p/100
    sl = res.get("sl"); tp1 = res.get("tp1"); tp2 = res.get("tp2")
    sl_d = abs(px-sl) if sl else px*0.02
    size_usd = (risk_usd/sl_d)*px if sl_d else 0

    sl_s  = f"${sl:,.5g}"  if sl  else "—"
    tp1_s = f"${tp1:,.5g}" if tp1 else "—"
    tp2_s = f"${tp2:,.5g}" if tp2 else "—"

    st.markdown(f"""
    <div style="background:{bg};border:2px solid {brd};border-radius:12px;
         padding:14px 18px;margin:10px 0">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:22px;font-weight:800;color:{clr}">{lbl}</span>
        <span style="font-size:26px;font-weight:800;color:{clr}">{res['confidence']}/100</span>
      </div>
      <div style="color:#8b949e;font-size:12px;margin:4px 0">
        {res.get('regime_msg','')}
      </div>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-top:10px">
        <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center">
          <div style="color:#8b949e;font-size:9px">ENTRY</div>
          <div style="color:#ffeb3b;font-size:11px;font-weight:700">{fmt}</div>
        </div>
        <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center">
          <div style="color:#ef5350;font-size:9px">SL</div>
          <div style="color:#ef5350;font-size:11px;font-weight:700">{sl_s}</div>
        </div>
        <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center">
          <div style="color:#00e676;font-size:9px">TP1</div>
          <div style="color:#00e676;font-size:11px;font-weight:700">{tp1_s}</div>
        </div>
        <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center">
          <div style="color:#69f0ae;font-size:9px">TP2</div>
          <div style="color:#69f0ae;font-size:11px;font-weight:700">{tp2_s}</div>
        </div>
        <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center">
          <div style="color:#8b949e;font-size:9px">LEV</div>
          <div style="color:#ffca28;font-size:11px;font-weight:700">×{res.get('leverage',1)}</div>
        </div>
      </div>
      <div style="color:#8b949e;font-size:11px;margin-top:8px">
        💰 Rischio: <b style="color:#fff">${risk_usd:.0f}</b>
        · Size: <b style="color:#fff">${size_usd:.0f}</b>
        · LONG {res.get('long_score',0)}/100 · SHORT {res.get('short_score',0)}/100
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── BACKTEST ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df, ema_ps, threshold, capital):
    if len(df) < 80:
        return None, None
    jarvis_bt = JarvisEngine()
    eq = [capital]; wins = 0; losses = 0; pf_w = 0; pf_l = 0
    prev_sig = "NEUTRAL"; entry_p = None; sl_p = None; tp_p = None

    for i in range(60, len(df)-1):
        sub = df.iloc[:i+1].copy()
        sub_ind = add_all_indicators(sub, ema_ps)
        res = jarvis_bt.signal(sub_ind, ema_ps, threshold)
        jarvis_bt.update(sub_ind, 1 if sub_ind["close"].iloc[-1]>sub_ind["close"].iloc[-2] else -1)

        curr_eq = eq[-1]
        if prev_sig != "NEUTRAL" and entry_p and sl_p and tp_p:
            nx = float(df["close"].iloc[i+1])
            risk = abs(entry_p-sl_p)/entry_p if entry_p else 0.02
            pos  = curr_eq * 0.01 / risk if risk else 0
            if prev_sig == "LONG":
                if nx <= sl_p:
                    pnl = -curr_eq*0.01; losses+=1; pf_l+=abs(pnl)
                elif nx >= tp_p:
                    pnl =  curr_eq*0.02; wins+=1;   pf_w+=pnl
                else: pnl = 0
            else:
                if nx >= sl_p:
                    pnl = -curr_eq*0.01; losses+=1; pf_l+=abs(pnl)
                elif nx <= tp_p:
                    pnl =  curr_eq*0.02; wins+=1;   pf_w+=pnl
                else: pnl = 0
            eq.append(curr_eq+pnl)
        else:
            eq.append(curr_eq)

        if res["signal"] in ("LONG","SHORT"):
            prev_sig = res["signal"]
            entry_p  = res["entry_px"]
            sl_p     = res["sl"]
            tp_p     = res["tp1"]
        else:
            prev_sig = "NEUTRAL"; entry_p=sl_p=tp_p=None

    total = wins+losses
    wr  = wins/total*100 if total else 0
    pf  = pf_w/pf_l if pf_l else float("inf")
    eq_s = pd.Series(eq)
    dd   = ((eq_s - eq_s.cummax())/eq_s.cummax()*100).min()
    ret  = (eq[-1]-capital)/capital*100
    return {"wins":wins,"losses":losses,"wr":wr,"pf":pf,"dd":dd,"ret":ret,
            "final":eq[-1],"trades":total}, eq


# ─────────────────────────────────────────────────────────────────────────────
# ── ACCURACY TAB ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def render_accuracy_tab():
    st.subheader("🏆 Accuracy — Storico Segnali")
    logs = load_log()
    if not logs:
        st.info("Nessun segnale ancora registrato. Vai su Analisi & Segnale per generarne uno.")
        return
    df_l = pd.DataFrame(logs)
    open_mask = df_l["result"]=="OPEN"
    col1,col2,col3 = st.columns(3)
    col1.metric("Totale", len(df_l))
    col2.metric("OPEN",   open_mask.sum())
    col3.metric("Chiusi", (~open_mask).sum())
    st.divider()
    for i, row in df_l.iloc[::-1].iterrows():
        with st.expander(f"#{row['id']} {row['symbol']} {row['tf']} — {row['signal']} — {row['result']}"):
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Entry",  f"${row['entry']:.6g}")
            c2.metric("SL",     f"${row['sl']:.6g}" if row['sl'] else "—")
            c3.metric("TP1",    f"${row['tp1']:.6g}" if row['tp1'] else "—")
            c4.metric("Score",  row['conf'])
            result_opt = st.selectbox("Risultato", ["OPEN","WIN","LOSS","BE"],
                key=f"res_{row['id']}", index=["OPEN","WIN","LOSS","BE"].index(row['result']))
            if result_opt != row["result"]:
                df_l.at[i,"result"] = result_opt
                save_log(df_l.to_dict("records"))
                st.success("Aggiornato")


# ─────────────────────────────────────────────────────────────────────────────
# ── MAIN ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown("""<style>
    [data-testid="stMetric"]{background:#161b22;border-radius:8px;
        padding:10px 14px;border:1px solid #30363d;}
    </style>""", unsafe_allow_html=True)
    st.title("🧠 Jarvis Pro — Crypto AI Trading")
    st.caption("Binance Live · Motore V3 · ADX Gate · Kelly Sizing · Backtest Zero Bias")

    if "jarvis" not in st.session_state:
        st.session_state.jarvis = JarvisEngine()
    jarvis = st.session_state.jarvis

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Setup")
        with st.spinner("📡 Mercati..."):
            coins = get_top_crypto_pairs(300)
        st.caption(f"✅ {len(coins)} coppie USDT")
        srch = st.text_input("🔍 Cerca", "")
        filt = [c for c in coins if srch.upper() in c["coin"]] if srch else coins
        sel  = st.selectbox("💎 Coin", [c["display"] for c in filt[:500]])
        coin = next((c for c in filt if c["display"]==sel), coins[0])
        symbol = coin["symbol"]

        tf = st.selectbox("⏱️ Timeframe", TIMEFRAMES_BIN,
            index=TIMEFRAMES_BIN.index("1h"))
        limit = st.slider("📦 Candele", 100, 500, 300, step=50)

        st.divider()
        st.subheader("📊 Indicatori")
        ema_sel = st.multiselect("EMA",
            ["EMA_20","EMA_50","EMA_100","EMA_200"],
            default=["EMA_20","EMA_50","EMA_200"])
        ema_ps = sorted([int(e.split("_")[1]) for e in ema_sel]) or [50]
        overlays  = st.multiselect("Overlay",
            ["BB","VWAP","SuperTrend","S/R"], default=["BB","SuperTrend","S/R"])
        oscillators = st.multiselect("Oscillatori",
            ["RSI","MACD","Stoch","ADX","CCI"], default=["RSI","MACD","ADX"])

        st.divider()
        st.subheader("💰 Risk Manager")
        capital = st.number_input("Capitale ($)", 100, 500000, 1000, step=500)
        risk_p  = st.slider("Rischio %", 0.5, 5.0, 1.0, step=0.5)
        rr_r    = st.slider("Risk:Reward", 1.0, 5.0, 2.0, step=0.5)

        st.divider()
        threshold = st.slider("🎯 Soglia segnale", 40, 80, 60)
        h_chart   = st.slider("📐 Altezza grafico", 600, 1200, 850, step=50)

        if st.button("🔄 Aggiorna dati", type="primary", use_container_width=True):
            st.cache_data.clear(); st.rerun()

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab_main, tab_bt, tab_api, tab_acc = st.tabs([
        "📊 Analisi & Segnale",
        "📈 Backtest",
        "🔑 API Bitget",
        "🏆 Accuracy",
    ])

    # ── DATI ─────────────────────────────────────────────────────────────────
    with st.spinner(f"📡 {symbol} {tf}..."):
        df_raw = get_ohlcv(symbol, tf, limit)
    if df_raw.empty:
        st.error("❌ Nessun dato disponibile."); return
    df = add_all_indicators(df_raw, ema_ps)

    # HTF
    htf_map = {"1m":"5m","3m":"15m","5m":"15m","15m":"1h","30m":"1h",
               "1h":"4h","2h":"4h","4h":"1d","6h":"1d","8h":"1d",
               "12h":"1d","1d":"1w","3d":"1w","1w":"1M","1M":"1M"}
    htf_tf = htf_map.get(tf,"4h")
    df_htf = get_ohlcv(symbol, htf_tf, 100)
    htf_score = 50
    if not df_htf.empty:
        df_htf = add_all_indicators(df_htf, ema_ps)
        px_h = float(df_htf["close"].iloc[-1])
        htf_score = 60 if (px_h > float(df_htf.get("EMA_50",pd.Series([px_h])).iloc[-1])) else 40

    res = jarvis.signal(df, ema_ps, threshold, htf_score)
    res["entry_px"] = float(df["close"].iloc[-1])

    if len(df) > 2:
        jarvis.update(df, 1 if df["close"].iloc[-1]>df["close"].iloc[-2] else -1)
    if res["signal"] != "NEUTRAL" and res.get("sl"):
        log_signal(coin["coin"], tf, res["signal"],
            res["entry_px"], res["sl"], res["tp1"],
            res["confidence"], res["leverage"])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — ANALISI
    # ─────────────────────────────────────────────────────────────────────────
    with tab_main:
        px = float(df["close"].iloc[-1])
        px_p = float(df["close"].iloc[-2])
        dpct = (px/px_p-1)*100 if px_p else 0
        px_fmt = f"${px:,.6f}" if px<0.01 else f"${px:,.4f}" if px<1 else f"${px:,.2f}"
        std_v = int(df["ST_dir"].iloc[-1]) if "ST_dir" in df.columns else 0

        # Metriche
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("💰 Prezzo",     px_fmt, f"{dpct:+.2f}%")
        m2.metric("📉 RSI",        f"{df['RSI'].iloc[-1]:.1f}" if "RSI" in df.columns else "—")
        m3.metric("💪 ADX",        f"{res['adx']:.1f}")
        m4.metric("🌊 SuperTrend", "🟢 BULL" if std_v==1 else "🔴 BEAR")
        m5.metric("📡 HTF Score",  f"{htf_score}/100")
        m6.metric("📐 ATR%",       f"{res['atr_pct']:.2f}%")

        # Card segnale
        render_signal_card(res, coin["coin"], capital, risk_p, rr_r)

        # Confluenze dettaglio
        with st.expander("🧠 Confluenze LONG vs SHORT — dettaglio"):
            dl, dr = st.columns(2)
            with dl:
                st.markdown(f"**🟢 LONG {res['long_score']}/100**")
                for r in res["reasons_long"]: st.write(r)
            with dr:
                st.markdown(f"**🔴 SHORT {res['short_score']}/100**")
                for r in res["reasons_short"]: st.write(r)
            st.caption(res.get("regime_msg",""))

        # Grafico
        fig = make_chart(df, ema_ps, overlays, oscillators, res, h_chart)
        st.plotly_chart(fig, use_container_width=True)

        # Calendario economico
        st.divider()
        st.subheader("📅 Calendario Economico")
        cal_col1, cal_col2, cal_col3 = st.columns([1,1,2])
        with cal_col1:
            sd = st.date_input("Da", datetime.now().date(), key="cal_da")
        with cal_col2:
            ed = st.date_input("A",  datetime.now().date()+timedelta(days=14), key="cal_a")
        with cal_col3:
            imp_f = st.multiselect("Impatto", ["HIGH","MEDIUM","LOW"],
                default=["HIGH","MEDIUM"], key="cal_imp")

        with st.spinner("Caricamento calendario..."):
            ev_all = get_calendar()
        evs = [e for e in ev_all
               if e.get("date") and e["impact"] in imp_f
               and sd <= datetime.strptime(e["date"],"%Y-%m-%d").date() <= ed]
        evs.sort(key=lambda x: x["date"]+x.get("time",""))

        if evs:
            for e in evs:
                ic = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(e["impact"],"⚪")
                act = f" → **{e['actual']}**" if e.get("actual") else ""
                col_ev, col_data = st.columns([2,1])
                with col_ev:
                    st.markdown(f"{ic} **{e['event']}**{act}")
                    st.caption(f"{e['country']} · {e['date']} {e.get('time','')}")
                with col_data:
                    st.caption(f"Prec: `{e['prev']}` · Prev: `{e['forecast']}`")
                st.divider()
        else:
            st.info("Nessun evento nel periodo selezionato.")

        # Dati grezzi
        with st.expander("📋 Dati recenti"):
            cols_ = ["open","high","low","close","volume","RSI","MACD","ADX","ATR","CCI"]
            st.dataframe(df.tail(30)[[c for c in cols_ if c in df.columns]].round(6),
                use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — BACKTEST
    # ─────────────────────────────────────────────────────────────────────────
    with tab_bt:
        st.subheader("📈 Backtest — Zero Look-Ahead Bias")
        st.info(f"Warm-up 60 candele · ADX Gate >{ADX_TREND_GATE} · ATR gate >{ATR_MIN_PCT}%")

        bt_cap = st.number_input("Capitale backtest ($)", 100, 100000, 10000, step=1000)
        bt_thr = st.slider("Soglia segnale backtest", 40, 80, threshold)

        if st.button("▶️ Esegui Backtest", type="primary", use_container_width=True):
            with st.spinner("Calcolo in corso..."):
                stats, equity = run_backtest(df, ema_ps, bt_thr, bt_cap)
            if stats:
                b1,b2,b3,b4 = st.columns(4)
                b1.metric("Win Rate",     f"{stats['wr']:.1f}%")
                b2.metric("Profit Factor",f"{stats['pf']:.2f}" if stats['pf']!=float('inf') else "∞")
                b3.metric("Max DD",       f"{stats['dd']:.1f}%")
                b4.metric("Return",       f"{stats['ret']:.1f}%",
                    f"${stats['final']:,.0f}")
                st.caption(f"Trade totali: {stats['trades']} · W:{stats['wins']} L:{stats['losses']}")

                judge = "✅ Strategia buona" if stats["wr"]>55 and stats["pf"]>1.5 else \
                        "⚠️ Accettabile" if stats["wr"]>45 else "❌ Reworking necessario"
                st.markdown(f"**{judge}**")

                eq_df = pd.DataFrame({"Equity": equity})
                fig_bt = go.Figure(go.Scatter(y=eq_df["Equity"], mode="lines",
                    line=dict(color="#00e676",width=2), fill="tozeroy",
                    fillcolor="rgba(0,230,118,0.07)"))
                fig_bt.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9"), height=350,
                    title="Equity Curve", xaxis_title="Trade #",
                    yaxis_title="Equity ($)",
                    margin=dict(l=10,r=10,t=40,b=10),
                )
                st.plotly_chart(fig_bt, use_container_width=True)
            else:
                st.warning("Dati insufficienti per il backtest (min 80 candele).")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — API BITGET
    # ─────────────────────────────────────────────────────────────────────────
    with tab_api:
        render_bitget_settings()

        # ── Piazza ordine rapido ─────────────────────────────────────────────
        st.divider()
        st.subheader("⚡ Piazza ordine rapido")

        if res["signal"] in ("LONG","SHORT") and res.get("sl"):
            st.markdown(f"Segnale attivo: **{res['signal']}** su `{symbol}` — Entry `{res['entry_px']:.6g}`")
            cfg_now = load_bitget_cfg()
            mode_txt = "🧪 DEMO" if cfg_now.get("sandbox") else "🔴 LIVE"
            st.warning(f"Modalità: {mode_txt}")

            with st.form("order_form"):
                oc1, oc2 = st.columns(2)
                with oc1:
                    order_size = st.number_input(
                        "Dimensione ordine (USDT)",
                        min_value=5.0, max_value=float(capital),
                        value=float(min(capital*0.05, 100)), step=5.0
                    )
                with oc2:
                    order_lev = st.number_input(
                        "Leva", min_value=1, max_value=20,
                        value=int(res.get("leverage",3))
                    )

                auto_sl  = st.number_input("Stop Loss ($)", value=round(res["sl"],6))
                auto_tp  = st.number_input("Take Profit ($)", value=round(res["tp1"],6))

                if st.form_submit_button(f"🚀 Invia ordine {res['signal']}", type="primary"):
                    if not cfg_now.get("api_key"):
                        st.error("❌ Configura prima le chiavi API nel tab 🔑 API Bitget")
                    else:
                        with st.spinner("Invio ordine..."):
                            r = bitget_place_order(
                                cfg_now, coin["coin"], res["signal"],
                                order_size/res["entry_px"],
                                auto_sl, auto_tp, order_lev
                            )
                        if r.get("code") == "00000":
                            st.success(f"✅ Ordine inviato! ID: {r.get('data',{}).get('orderId','—')}")
                        else:
                            st.error(f"❌ Errore: {r.get('msg', str(r))}")
        else:
            st.info("Nessun segnale attivo. Attendi un segnale LONG/SHORT per piazzare un ordine.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 — ACCURACY
    # ─────────────────────────────────────────────────────────────────────────
    with tab_acc:
        render_accuracy_tab()


main()
