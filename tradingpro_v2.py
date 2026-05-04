import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json, os, time, hmac, hashlib, warnings
from collections import deque
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
ADX_TREND_GATE = 18
ATR_MIN_PCT    = 0.15
SIGNAL_LOG     = "jarvis_signal_log.json"
BITGET_CFG     = "bitget_cfg.json"
TIMEFRAMES_BIN = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"]

BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

# ─────────────────────────────────────────────────────────────────────────────
# BINANCE — LISTA COIN (multi-endpoint + fallback hardcoded)
# ─────────────────────────────────────────────────────────────────────────────

HARDCODED_COINS = [
    "BTC","ETH","BNB","SOL","XRP","DOGE","ADA","AVAX","TRX","SHIB",
    "DOT","LINK","MATIC","LTC","UNI","ATOM","APT","OP","ARB","NEAR",
    "FTM","AAVE","INJ","IMX","FIL","HBAR","STX","RUNE","LDO","CRV",
    "SNX","COMP","MKR","1INCH","SUSHI","CHZ","FLOW","ROSE","ANKR",
    "SAND","MANA","ENS","GRT","OCEAN","FET","KAVA","ONE","ZIL","BAND",
    "VET","ICP","ALGO","EGLD","THETA","AXS","GALA","ENJ","BAT","ZRX",
]

def _binance_get(path: str, params: dict = None, timeout: int = 8):
    """Prova tutti gli endpoint Binance in sequenza."""
    for base in BINANCE_ENDPOINTS:
        try:
            r = requests.get(base + path, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    return None

@st.cache_data(ttl=3600)
def get_top_crypto_pairs(limit: int = 200):
    data = _binance_get("/api/v3/ticker/24hr", timeout=10)
    if data:
        usdt = [d for d in data if isinstance(d, dict) and d.get("symbol","").endswith("USDT")]
        usdt.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
        result = []
        for p in usdt[:limit]:
            coin = p["symbol"].replace("USDT","")
            price = float(p.get("lastPrice", 0))
            vol   = float(p.get("quoteVolume", 0))
            result.append({
                "symbol":  p["symbol"],
                "coin":    coin,
                "price":   price,
                "volume":  vol,
                "display": f"{coin} — ${price:,.4f}",
            })
        if result:
            return result

    # Fallback hardcoded
    result = []
    for coin in HARDCODED_COINS:
        result.append({
            "symbol":  f"{coin}USDT",
            "coin":    coin,
            "price":   0.0,
            "volume":  0.0,
            "display": f"{coin} — (dati live non disp.)",
        })
    return result

@st.cache_data(ttl=45)
def get_ohlcv(symbol: str, interval: str = "1h", limit: int = 300):
    data = _binance_get("/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=12)
    if not data or not isinstance(data, list) or len(data) < 2:
        return pd.DataFrame()
    try:
        df = pd.DataFrame(data, columns=[
            "ts","open","high","low","close","volume",
            "cts","qav","trades","tbbase","tbquote","ignore"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["open","high","low","close"], inplace=True)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df[["open","high","low","close","volume"]]
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# INDICATORI TECNICI
# ─────────────────────────────────────────────────────────────────────────────

def ema(s, p):   return s.ewm(span=int(p), adjust=False).mean()
def sma(s, p):   return s.rolling(int(p)).mean()

def rsi14(s, p=14):
    d  = s.diff()
    g  = d.where(d > 0, 0.0).ewm(alpha=1/p, adjust=False).mean()
    l  = (-d.where(d < 0, 0.0)).ewm(alpha=1/p, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd_ind(s, f=12, sl=26, sig=9):
    m  = ema(s, f) - ema(s, sl)
    sg = ema(m, sig)
    return m, sg, m - sg

def boll(s, p=20, k=2):
    mid = sma(s, p); std_ = s.rolling(p).std()
    return mid + k*std_, mid - k*std_, mid

def atr14(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def adx_full(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    up  = h.diff(); dn = -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    atr_ = atr14(df, p).replace(0, np.nan)
    pdi  = 100 * pdm.ewm(alpha=1/p, adjust=False).mean() / atr_
    ndi  = 100 * ndm.ewm(alpha=1/p, adjust=False).mean() / atr_
    dx   = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1/p, adjust=False).mean(), pdi, ndi

def stoch14(df, k=14, d=3):
    lo  = df["low"].rolling(k).min()
    hi  = df["high"].rolling(k).max()
    pct = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return pct, pct.rolling(d).mean()

def vwap_calc(df):
    tp      = (df["high"] + df["low"] + df["close"]) / 3
    cum_tpv = (tp * df["volume"]).cumsum()
    cum_v   = df["volume"].cumsum().replace(0, np.nan)
    return cum_tpv / cum_v

def cci20(df, p=20):
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    ma  = tp.rolling(p).mean()
    md  = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).replace(0, np.nan)
    return (tp - ma) / (0.015 * md)

def supertrend(df, p=10, mult=3.0):
    atr_ = atr14(df, p)
    hl2  = (df["high"] + df["low"]) / 2
    up   = hl2 + mult * atr_
    dn   = hl2 - mult * atr_
    direction = [1] * len(df)
    st_up = dn.copy(); st_dn = up.copy()
    for i in range(1, len(df)):
        st_up.iloc[i] = max(dn.iloc[i], st_up.iloc[i-1]) \
            if df["close"].iloc[i-1] > st_up.iloc[i-1] else dn.iloc[i]
        st_dn.iloc[i] = min(up.iloc[i], st_dn.iloc[i-1]) \
            if df["close"].iloc[i-1] < st_dn.iloc[i-1] else up.iloc[i]
        if direction[i-1] == -1:
            direction[i] = 1 if df["close"].iloc[i] > st_dn.iloc[i] else -1
        else:
            direction[i] = -1 if df["close"].iloc[i] < st_up.iloc[i] else 1
    dir_s  = pd.Series(direction, index=df.index)
    line_s = pd.Series(np.where(dir_s == 1, st_up, st_dn), index=df.index)
    return line_s, dir_s

def pivot_sr(df, window=10):
    levels = []
    for i in range(window, len(df) - window):
        if df["high"].iloc[i] == df["high"].iloc[i-window:i+window+1].max():
            levels.append(("R", float(df["high"].iloc[i])))
        if df["low"].iloc[i] == df["low"].iloc[i-window:i+window+1].min():
            levels.append(("S", float(df["low"].iloc[i])))
    return levels[-12:]

def add_all_indicators(df, ema_ps=(20, 50, 200)):
    if df.empty or len(df) < 30:
        return df
    df = df.copy()
    c  = df["close"]
    for p in ema_ps:
        df[f"EMA_{p}"] = ema(c, p)
    df["RSI"]              = rsi14(c)
    df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd_ind(c)
    df["BB_up"], df["BB_lo"], df["BB_mid"]       = boll(c)
    df["ATR"]              = atr14(df)
    df["ADX"], df["DI_plus"], df["DI_minus"]     = adx_full(df)
    df["STOCH_K"], df["STOCH_D"]                 = stoch14(df)
    df["VWAP"]             = vwap_calc(df)
    df["CCI"]              = cci20(df)
    df["VOLMA"]            = sma(df["volume"], 20)
    df["ST_line"], df["ST_dir"]                  = supertrend(df)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MOTORE JARVIS V3
# ─────────────────────────────────────────────────────────────────────────────

class JarvisEngine:
    def __init__(self, k=12, hist=500):
        self.k          = k
        self.feat_hist  = deque(maxlen=hist)
        self.label_hist = deque(maxlen=hist)

    def _features(self, df):
        rsi_v = float(df["RSI"].iloc[-1])  if "RSI" in df.columns else 50.0
        cci_v = float(df["CCI"].iloc[-1])  if "CCI" in df.columns else 0.0
        rsi_n = np.clip((rsi_v - 30) / 40, 0, 1)
        cci_n = np.clip((cci_v + 200) / 400, 0, 1)
        return np.array([rsi_n, cci_n])

    def update(self, df, label):
        self.feat_hist.append(self._features(df))
        self.label_hist.append(label)

    def knn_predict(self, df):
        if len(self.feat_hist) < self.k:
            return 0
        cur   = self._features(df)
        X     = np.array(list(self.feat_hist))
        y     = np.array(list(self.label_hist))
        dists = np.sqrt(((np.log1p(X) - np.log1p(cur)) ** 2).sum(axis=1))
        idx   = np.argsort(dists)[: self.k]
        return 1 if y[idx].sum() > 0 else -1

    def signal(self, df, ema_ps, threshold=60, htf_score=50):
        if df.empty or len(df) < 50:
            return self._neutral("Dati insufficienti")

        # ── Valori base ──────────────────────────────────────────────────────
        px    = float(df["close"].iloc[-1])
        adx_v = float(df["ADX"].iloc[-1])  if "ADX" in df.columns else 0.0
        atr_v = float(df["ATR"].iloc[-1])  if "ATR" in df.columns else px * 0.01
        atr_p = atr_v / px * 100 if px else 0.0

        # ── Regime gate (soglie abbassate) ───────────────────────────────────
        if adx_v < ADX_TREND_GATE:
            return self._neutral(f"ADX {adx_v:.1f} < {ADX_TREND_GATE} — laterale")
        if atr_p < ATR_MIN_PCT:
            return self._neutral(f"ATR {atr_p:.2f}% < {ATR_MIN_PCT}% — piatto")

        long_s = 0; short_s = 0; rl = []; rs = []

        # EMA stack (18)
        ev = {p: float(df[f"EMA_{p}"].iloc[-1]) for p in ema_ps if f"EMA_{p}" in df.columns}
        srt = sorted(ev)
        if len(srt) >= 2:
            if all(ev[srt[i]] > ev[srt[i+1]] for i in range(len(srt)-1)) and px > ev[srt[0]]:
                long_s += 18; rl.append("✅ EMA stack bullish perfetto")
            elif all(ev[srt[i]] < ev[srt[i+1]] for i in range(len(srt)-1)) and px < ev[srt[0]]:
                short_s += 18; rs.append("✅ EMA stack bearish perfetto")
            elif px > ev.get(srt[0], px):
                long_s += 8; rl.append("⚠️ EMA parziale bullish")
            else:
                short_s += 8; rs.append("⚠️ EMA parziale bearish")

        # SuperTrend (22)
        if "ST_dir" in df.columns:
            std = int(df["ST_dir"].iloc[-1])
            if   std ==  1: long_s  += 22; rl.append("✅ SuperTrend BULL")
            elif std == -1: short_s += 22; rs.append("✅ SuperTrend BEAR")

        # kNN (12)
        knn = self.knn_predict(df)
        if   knn ==  1: long_s  += 12; rl.append("✅ kNN rialzista")
        elif knn == -1: short_s += 12; rs.append("✅ kNN ribassista")

        # CVD (10)
        if "volume" in df.columns:
            delta = df["volume"] * np.sign(df["close"].diff().fillna(0))
            cvd   = delta.cumsum()
            if df["close"].iloc[-1] <= df["close"].rolling(20).min().iloc[-1] * 1.01 \
               and cvd.iloc[-1] > cvd.rolling(20).min().iloc[-1]:
                long_s += 10; rl.append("✅ CVD assorbimento long")
            elif df["close"].iloc[-1] >= df["close"].rolling(20).max().iloc[-1] * 0.99 \
               and cvd.iloc[-1] < cvd.rolling(20).max().iloc[-1]:
                short_s += 10; rs.append("✅ CVD assorbimento short")

        # HTF (10)
        if   htf_score >= 60: long_s  += 10; rl.append(f"✅ HTF {htf_score}/100 bullish")
        elif htf_score <= 40: short_s += 10; rs.append(f"✅ HTF {htf_score}/100 bearish")

        # MACD (6)
        if "MACD" in df.columns and "MACD_sig" in df.columns:
            if float(df["MACD"].iloc[-1]) > float(df["MACD_sig"].iloc[-1]):
                long_s += 6; rl.append("✅ MACD positivo")
            else:
                short_s += 6; rs.append("✅ MACD negativo")

        # RSI (6)
        rsi_v = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 50.0
        if   rsi_v < 40: long_s  += 6; rl.append(f"✅ RSI {rsi_v:.0f} oversold")
        elif rsi_v > 60: short_s += 6; rs.append(f"✅ RSI {rsi_v:.0f} overbought")

        # ADX boost (8)
        if adx_v >= 28:
            if long_s > short_s:   long_s  += 8; rl.append(f"✅ ADX {adx_v:.0f} trend forte")
            elif short_s > long_s: short_s += 8; rs.append(f"✅ ADX {adx_v:.0f} trend forte")

        # Volume (8)
        if "VOLMA" in df.columns:
            vc = float(df["volume"].iloc[-1]); vm = float(df["VOLMA"].iloc[-1])
            if vc > vm * 1.2:
                if long_s >= short_s:  long_s  += 8; rl.append("✅ Volume sopra media")
                else:                  short_s += 8; rs.append("✅ Volume sopra media")

        # Normalizza (max raw ~100)
        long_s  = min(100, long_s)
        short_s = min(100, short_s)

        # Indecisione
        if long_s >= threshold and short_s >= threshold:
            return {**self._neutral("Segnali contrari — mercato indeciso"),
                    "long_score": long_s, "short_score": short_s,
                    "reasons_long": rl, "reasons_short": rs,
                    "adx": adx_v, "atr_pct": atr_p, "rsi": rsi_v}

        # Direzione
        if long_s >= threshold and long_s > short_s:
            signal = "LONG";  conf = long_s
        elif short_s >= threshold and short_s > long_s:
            signal = "SHORT"; conf = short_s
        else:
            return {**self._neutral(f"Score L={long_s} S={short_s} < soglia {threshold}"),
                    "long_score": long_s, "short_score": short_s,
                    "reasons_long": rl, "reasons_short": rs,
                    "adx": adx_v, "atr_pct": atr_p, "rsi": rsi_v}

        # Livelli con slippage
        slip  = px * 0.0004
        sl_d  = atr_v * 1.5
        if signal == "LONG":
            entry = px + slip; sl = px - sl_d
            tp1   = px + sl_d * 2.0; tp2 = px + sl_d * 3.236
        else:
            entry = px - slip; sl = px + sl_d
            tp1   = px - sl_d * 2.0; tp2 = px - sl_d * 3.236

        # Leva Kelly
        kelly    = max(0.05, min(0.25, (conf/100*1.5 - (1-conf/100)) / 1.5 * 0.25))
        risk_frc = abs(px - sl) / px if px else 0.02
        leverage = int(np.clip(kelly / risk_frc if risk_frc else 3, 1, 10))
        if adx_v > 30:  leverage = min(leverage + 1, 10)
        if atr_p > 2.5: leverage = max(leverage - 1, 1)

        regime = "TREND" if adx_v >= 25 else "LATERALE"
        return {
            "signal": signal, "confidence": conf,
            "long_score": long_s, "short_score": short_s,
            "reasons_long": rl, "reasons_short": rs,
            "entry_px": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
            "leverage": leverage, "adx": adx_v, "atr_pct": atr_p, "rsi": rsi_v,
            "regime_msg": f"{regime} · ADX {adx_v:.1f} · ATR {atr_p:.2f}%",
        }

    @staticmethod
    def _neutral(msg):
        return {
            "signal": "NEUTRAL", "confidence": 0,
            "long_score": 0, "short_score": 0,
            "reasons_long": [], "reasons_short": [f"⚠️ {msg}"],
            "entry_px": 0, "sl": None, "tp1": None, "tp2": None,
            "leverage": 1, "adx": 0, "atr_pct": 0, "rsi": 50,
            "regime_msg": msg,
        }

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL LOG
# ─────────────────────────────────────────────────────────────────────────────

def load_log():
    try:
        if os.path.exists(SIGNAL_LOG):
            return json.load(open(SIGNAL_LOG))
    except Exception:
        pass
    return []

def save_log(logs):
    try:
        json.dump(logs, open(SIGNAL_LOG, "w"), indent=2)
    except Exception:
        pass

def log_signal(symbol, tf, signal, entry, sl, tp1, conf, lev):
    logs = load_log()
    logs.append({
        "id": len(logs) + 1, "ts": datetime.now().isoformat(),
        "symbol": symbol, "tf": tf, "signal": signal,
        "entry": round(float(entry), 8),
        "sl":    round(float(sl),    8) if sl  else None,
        "tp1":   round(float(tp1),   8) if tp1 else None,
        "conf":  conf, "leverage": lev, "result": "OPEN",
    })
    save_log(logs)

# ─────────────────────────────────────────────────────────────────────────────
# GRAFICO
# ─────────────────────────────────────────────────────────────────────────────

def make_chart(df, ema_ps, overlays, oscillators, res, height=820):
    n_sub  = max(len(oscillators), 0)
    rh     = [0.52] + [round(0.48 / n_sub, 3)] * n_sub if n_sub else [1.0]
    rh_n   = [r / sum(rh) for r in rh]
    titles = ["📈 Prezzo"] + [f"📊 {o}" for o in oscillators]

    fig = make_subplots(
        rows=1 + n_sub, cols=1, shared_xaxes=True,
        vertical_spacing=0.025, row_heights=rh_n,
        subplot_titles=titles,
    )

    # Candele
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"],  close=df["close"],
        increasing_fillcolor="#00897b", increasing_line_color="#00897b",
        decreasing_fillcolor="#c62828", decreasing_line_color="#c62828",
        name="OHLC", showlegend=False,
    ), row=1, col=1)

    # Volume
    vol_c = ["#00897b" if df["close"].iloc[i] >= df["open"].iloc[i] else "#c62828"
             for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], marker_color=vol_c,
        name="Volume", opacity=0.30, showlegend=False, yaxis="y2",
    ), row=1, col=1)

    # EMAs
    ema_colors = {20:"#ffca28", 50:"#ff7043", 100:"#ab47bc", 200:"#ef5350"}
    for p in ema_ps:
        col_ = f"EMA_{p}"
        if col_ in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_], name=f"EMA{p}",
                line=dict(color=ema_colors.get(p, "#aaa"), width=1.4),
            ), row=1, col=1)

    # Bollinger
    if "BB" in overlays and "BB_up" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], name="BB Up",
            line=dict(color="#78909c", width=1, dash="dot"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lo"], name="BB Lo",
            line=dict(color="#78909c", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(120,144,156,0.07)", showlegend=False), row=1, col=1)

    # VWAP
    if "VWAP" in overlays and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
            line=dict(color="#f48fb1", width=1.3, dash="dash")), row=1, col=1)

    # SuperTrend
    if "SuperTrend" in overlays and "ST_line" in df.columns:
        bull = df["ST_dir"] == 1; bear = df["ST_dir"] == -1
        if bull.any():
            fig.add_trace(go.Scatter(x=df.index[bull], y=df["ST_line"][bull],
                name="ST Bull", line=dict(color="#00e676", width=2), mode="lines"), row=1, col=1)
        if bear.any():
            fig.add_trace(go.Scatter(x=df.index[bear], y=df["ST_line"][bear],
                name="ST Bear", line=dict(color="#ef5350", width=2), mode="lines"), row=1, col=1)

    # S/R
    if "S/R" in overlays:
        shown = set()
        for kind, lvl in pivot_sr(df):
            if lvl in shown: continue
            shown.add(lvl)
            fig.add_hline(y=lvl, row=1, col=1,
                line=dict(color="#ffd54f" if kind=="R" else "#80cbc4", width=1, dash="dot"),
                annotation_text=kind, annotation_position="right")

    # Freccia segnale
    sig = res.get("signal", "NEUTRAL")
    if sig in ("LONG", "SHORT") and res.get("sl"):
        lx = df.index[-1]; ly = float(df["close"].iloc[-1])
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly], mode="markers+text",
            marker=dict(symbol="triangle-up" if sig=="LONG" else "triangle-down",
                        size=18, color="#00e676" if sig=="LONG" else "#ef5350"),
            text=[sig], textposition="top center", name=sig, showlegend=False,
        ), row=1, col=1)
        # SL/TP lines
        fig.add_hline(y=res["sl"],  row=1, col=1,
            line=dict(color="#ef5350", width=1, dash="dash"),
            annotation_text="SL", annotation_position="right")
        fig.add_hline(y=res["tp1"], row=1, col=1,
            line=dict(color="#00e676", width=1, dash="dash"),
            annotation_text="TP1", annotation_position="right")

    # Oscillatori subplot
    for ri, osc in enumerate(oscillators, start=2):
        if osc == "RSI" and "RSI" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                line=dict(color="#ce93d8", width=1.5)), row=ri, col=1)
            for y_v, c_ in [(70,"#ef5350"),(30,"#00e676"),(50,"#444")]:
                fig.add_hline(y=y_v, line_color=c_, line_dash="dot", row=ri, col=1)
            fig.update_yaxes(range=[0,100], row=ri, col=1)

        elif osc == "MACD" and "MACD" in df.columns:
            hc = ["#00897b" if v >= 0 else "#c62828" for v in df["MACD_hist"]]
            fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"],
                marker_color=hc, name="MACD Hist", opacity=0.7), row=ri, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                line=dict(color="#64b5f6", width=1.3)), row=ri, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_sig"], name="Signal",
                line=dict(color="#ef9a9a", width=1.3)), row=ri, col=1)

        elif osc == "Stoch" and "STOCH_K" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_K"], name="%K",
                line=dict(color="#4fc3f7", width=1.3)), row=ri, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_D"], name="%D",
                line=dict(color="#f48fb1", width=1.3)), row=ri, col=1)
            for y_v, c_ in [(80,"#ef5350"),(20,"#00e676")]:
                fig.add_hline(y=y_v, line_color=c_, line_dash="dot", row=ri, col=1)
            fig.update_yaxes(range=[0,100], row=ri, col=1)

        elif osc == "ADX" and "ADX" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX",
                line=dict(color="#ffca28", width=1.5)), row=ri, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["DI_plus"], name="DI+",
                line=dict(color="#00e676", width=1, dash="dot")), row=ri, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["DI_minus"], name="DI-",
                line=dict(color="#ef5350", width=1, dash="dot")), row=ri, col=1)
            fig.add_hline(y=ADX_TREND_GATE, line_color="#888", line_dash="dash", row=ri, col=1)

        elif osc == "CCI" and "CCI" in df.columns:
            cc = ["#00897b" if v >= 0 else "#c62828" for v in df["CCI"]]
            fig.add_trace(go.Bar(x=df.index, y=df["CCI"],
                marker_color=cc, name="CCI", opacity=0.7), row=ri, col=1)
            for y_v, c_ in [(100,"#ef5350"),(-100,"#00e676")]:
                fig.add_hline(y=y_v, line_color=c_, line_dash="dot", row=ri, col=1)

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", size=11), height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified", xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    showticklabels=False, range=[0, df["volume"].max() * 5]),
    )
    for i in range(1, 2 + n_sub):
        fig.update_xaxes(gridcolor="#21262d", row=i, col=1)
        fig.update_yaxes(gridcolor="#21262d", row=i, col=1)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL CARD
# ─────────────────────────────────────────────────────────────────────────────

def render_signal_card(res, capital, risk_p):
    sig = res["signal"]
    bg, brd, lbl, clr = {
        "LONG":    ("#003300","#00e676","🟢 LONG",  "#00e676"),
        "SHORT":   ("#330000","#ef5350","🔴 SHORT", "#ef5350"),
        "NEUTRAL": ("#1a1a1a","#555",   "⚪ NEUTRAL","#aaa"),
    }.get(sig, ("#1a1a1a","#555","⚪ NEUTRAL","#aaa"))

    px    = res.get("entry_px", 0)
    sl    = res.get("sl");  tp1 = res.get("tp1"); tp2 = res.get("tp2")
    px_f  = f"${px:,.6f}" if px < 0.01 else f"${px:,.4f}" if px < 1 else f"${px:,.2f}"
    sl_f  = f"${sl:,.5g}"  if sl  else "—"
    tp1_f = f"${tp1:,.5g}" if tp1 else "—"
    tp2_f = f"${tp2:,.5g}" if tp2 else "—"

    risk_usd = capital * risk_p / 100
    sl_d     = abs(px - sl) if sl else px * 0.02
    size_usd = (risk_usd / sl_d) * px if sl_d and px else 0

    st.markdown(f"""
    <div style="background:{bg};border:2px solid {brd};border-radius:12px;
         padding:14px 18px;margin:10px 0">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:22px;font-weight:800;color:{clr}">{lbl}</span>
        <span style="font-size:26px;font-weight:800;color:{clr}">{res['confidence']}/100</span>
      </div>
      <div style="color:#8b949e;font-size:12px;margin:4px 0">{res.get('regime_msg','')}</div>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-top:10px">
        <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center">
          <div style="color:#8b949e;font-size:9px">ENTRY</div>
          <div style="color:#ffeb3b;font-size:11px;font-weight:700">{px_f}</div>
        </div>
        <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center">
          <div style="color:#ef5350;font-size:9px">SL</div>
          <div style="color:#ef5350;font-size:11px;font-weight:700">{sl_f}</div>
        </div>
        <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center">
          <div style="color:#00e676;font-size:9px">TP1</div>
          <div style="color:#00e676;font-size:11px;font-weight:700">{tp1_f}</div>
        </div>
        <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center">
          <div style="color:#69f0ae;font-size:9px">TP2</div>
          <div style="color:#69f0ae;font-size:11px;font-weight:700">{tp2_f}</div>
        </div>
        <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center">
          <div style="color:#8b949e;font-size:9px">LEV</div>
          <div style="color:#ffca28;font-size:11px;font-weight:700">×{res.get('leverage',1)}</div>
        </div>
      </div>
      <div style="color:#8b949e;font-size:11px;margin-top:8px">
        💰 Rischio: <b style="color:#fff">${risk_usd:.0f}</b>
        · Posizione: <b style="color:#fff">${size_usd:.0f}</b>
        · L:{res.get('long_score',0)} S:{res.get('short_score',0)}
      </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df, ema_ps, threshold, capital):
    if len(df) < 80:
        return None, None
    engine = JarvisEngine()
    eq     = [float(capital)]
    wins   = losses = 0
    pf_w   = pf_l = 0.0
    prev   = "NEUTRAL"; e_p = sl_p = tp_p = None

    for i in range(60, len(df) - 1):
        sub = add_all_indicators(df.iloc[: i+1].copy(), ema_ps)
        if sub.empty: continue
        r   = engine.signal(sub, ema_ps, threshold)
        engine.update(sub, 1 if sub["close"].iloc[-1] > sub["close"].iloc[-2] else -1)
        cur_eq = eq[-1]

        if prev != "NEUTRAL" and e_p and sl_p and tp_p:
            nx   = float(df["close"].iloc[i+1])
            risk = abs(e_p - sl_p) / e_p if e_p else 0.02
            if prev == "LONG":
                if   nx <= sl_p: pnl = -cur_eq * 0.01; losses += 1; pf_l += abs(pnl)
                elif nx >= tp_p: pnl =  cur_eq * 0.02; wins   += 1; pf_w += pnl
                else: pnl = 0.0
            else:
                if   nx >= sl_p: pnl = -cur_eq * 0.01; losses += 1; pf_l += abs(pnl)
                elif nx <= tp_p: pnl =  cur_eq * 0.02; wins   += 1; pf_w += pnl
                else: pnl = 0.0
            eq.append(cur_eq + pnl)
        else:
            eq.append(cur_eq)

        if r["signal"] in ("LONG","SHORT"):
            prev = r["signal"]; e_p = r["entry_px"]; sl_p = r["sl"]; tp_p = r["tp1"]
        else:
            prev = "NEUTRAL"; e_p = sl_p = tp_p = None

    total = wins + losses
    wr    = wins / total * 100 if total else 0
    pf    = pf_w / pf_l if pf_l else float("inf")
    eq_s  = pd.Series(eq)
    dd    = ((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min()
    ret   = (eq[-1] - capital) / capital * 100
    return {"wins": wins, "losses": losses, "wr": wr, "pf": pf,
            "dd": dd, "ret": ret, "final": eq[-1], "trades": total}, eq

# ─────────────────────────────────────────────────────────────────────────────
# CALENDARIO ECONOMICO
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_calendar():
    try:
        r = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=8)
        raw = r.json()
        result = []
        for e in raw:
            dt = e.get("date","")
            try:
                parsed = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S%z")
                ds = parsed.strftime("%Y-%m-%d"); ts = parsed.strftime("%H:%M")
            except Exception:
                ds = datetime.now().strftime("%Y-%m-%d"); ts = ""
            result.append({
                "date": ds, "time": ts,
                "country": e.get("country",""),
                "event":   e.get("title",""),
                "impact":  e.get("impact","LOW").upper(),
                "prev":    e.get("previous","—"),
                "forecast":e.get("forecast","—"),
                "actual":  e.get("actual",""),
            })
        if result: return result
    except Exception:
        pass
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
            (14,"14:30","🇺🇸","Jobless Claims","MEDIUM","220K","215K"),
        ]
    ]

# ─────────────────────────────────────────────────────────────────────────────
# BITGET API
# ─────────────────────────────────────────────────────────────────────────────

def load_bitget_cfg():
    try:
        if os.path.exists(BITGET_CFG):
            return json.load(open(BITGET_CFG))
    except Exception:
        pass
    return {"api_key":"","api_secret":"","passphrase":"","sandbox":True}

def save_bitget_cfg(cfg):
    try: json.dump(cfg, open(BITGET_CFG,"w"), indent=2)
    except Exception: pass

def _bg_sign(secret, ts, method, path, body=""):
    msg = ts + method.upper() + path + body
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest().hex()

def _bg_headers(cfg, method, path, body=""):
    ts  = str(int(time.time() * 1000))
    sig = _bg_sign(cfg["api_secret"], ts, method, path, body)
    return {
        "ACCESS-KEY":        cfg["api_key"],
        "ACCESS-SIGN":       sig,
        "ACCESS-TIMESTAMP":  ts,
        "ACCESS-PASSPHRASE": cfg["passphrase"],
        "Content-Type":      "application/json",
        "locale":            "en-US",
    }

BG_BASE = "https://api.bitget.com"

def bitget_test(cfg):
    try:
        path = "/api/v2/account/info"
        r = requests.get(BG_BASE+path, headers=_bg_headers(cfg,"GET",path), timeout=8)
        d = r.json()
        if d.get("code")=="00000":
            return True, f"UserID {d.get('data',{}).get('userId','—')}"
        return False, d.get("msg","Errore sconosciuto")
    except Exception as e:
        return False, str(e)

def bitget_balance(cfg):
    try:
        path = "/api/v2/mix/account/accounts?productType=USDT-FUTURES"
        r = requests.get(BG_BASE+path, headers=_bg_headers(cfg,"GET",path), timeout=8)
        d = r.json()
        if d.get("code")=="00000":
            out={}
            for item in d.get("data",[]):
                coin=item.get("marginCoin","USDT")
                out[coin]={"available":item.get("available","0"),
                           "equity":item.get("equity","0"),
                           "unrealized":item.get("unrealizedPL","0")}
            return out
    except Exception: pass
    return None

def bitget_positions(cfg):
    try:
        path="/api/v2/mix/position/all-position?productType=USDT-FUTURES&marginCoin=USDT"
        r = requests.get(BG_BASE+path, headers=_bg_headers(cfg,"GET",path), timeout=8)
        d = r.json()
        if d.get("code")=="00000":
            rows=[]
            for p in d.get("data",[]):
                size=float(p.get("total",0))
                if size==0: continue
                rows.append({
                    "Simbolo":   p.get("symbol",""),
                    "Dir":       "🟢 LONG" if p.get("holdSide")=="long" else "🔴 SHORT",
                    "Size":      size,
                    "Entry ($)": float(p.get("openPriceAvg",0)),
                    "Mark ($)":  float(p.get("markPrice",0)),
                    "PnL ($)":   float(p.get("unrealizedPL",0)),
                    "Leva":      p.get("leverage","—"),
                })
            return rows
    except Exception: pass
    return []

def bitget_place_order(cfg, symbol, side, size, sl, tp, leverage=5):
    try:
        path="/api/v2/mix/order/place-order"
        body_d={
            "symbol":f"{symbol}USDT","productType":"USDT-FUTURES",
            "marginMode":"isolated","marginCoin":"USDT",
            "size":str(round(size,4)),
            "side":"buy" if side=="LONG" else "sell",
            "tradeSide":"open","orderType":"market",
            "presetStopLossPrice":str(round(sl,6)),
            "presetStopSurplusPrice":str(round(tp,6)),
        }
        body=json.dumps(body_d)
        r=requests.post(BG_BASE+path,headers=_bg_headers(cfg,"POST",path,body),
                        data=body,timeout=10)
        return r.json()
    except Exception as e:
        return {"error":str(e)}

def render_bitget_settings(res, coin_sym, capital):
    st.subheader("🔑 Configurazione API Bitget")
    st.markdown("""
    <div style='background:#161b22;border:1px solid #30363d;border-left:4px solid #f0883e;
         border-radius:8px;padding:12px 16px;margin-bottom:16px'>
      <b style='color:#f0883e'>⚠️ Sicurezza</b><br>
      <span style='color:#8b949e;font-size:13px'>
      Le chiavi sono salvate localmente in <code>bitget_cfg.json</code>.
      Usa chiavi con soli permessi <b>Futures Read + Order</b> e IP whitelist attiva.
      </span>
    </div>""", unsafe_allow_html=True)

    cfg = load_bitget_cfg()
    with st.form("bitget_form"):
        c1, c2 = st.columns(2)
        with c1:
            api_key    = st.text_input("API Key",    value=cfg.get("api_key",""),    placeholder="Incolla API Key")
            passphrase = st.text_input("Passphrase", value=cfg.get("passphrase",""), type="password")
        with c2:
            api_secret = st.text_input("API Secret", value=cfg.get("api_secret",""), type="password")
            sandbox    = st.toggle("🧪 Modalità DEMO", value=cfg.get("sandbox",True))
        st.markdown("**Permessi consigliati:** ✅ Futures Read · ✅ Futures Order · ❌ Withdraw mai")
        if st.form_submit_button("💾 Salva", type="primary"):
            save_bitget_cfg({"api_key":api_key.strip(),"api_secret":api_secret.strip(),
                             "passphrase":passphrase.strip(),"sandbox":sandbox})
            st.success("✅ Salvato!")

    st.divider()
    cfg_now = load_bitget_cfg()
    mode_txt = "🧪 DEMO" if cfg_now.get("sandbox") else "🔴 LIVE"
    st.info(f"Modalità: **{mode_txt}**")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔌 Test connessione", use_container_width=True):
            if not cfg_now.get("api_key"):
                st.error("Nessuna chiave configurata.")
            else:
                ok, msg = bitget_test(cfg_now)
                (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {msg}")
    with col_b:
        if st.button("💰 Saldo Futures", use_container_width=True):
            bal = bitget_balance(cfg_now)
            if bal:
                for asset, info in bal.items():
                    st.metric(asset, f"${float(info['available']):,.2f}",
                              f"Equity ${float(info['equity']):,.2f}")
            else:
                st.warning("Impossibile recuperare il saldo.")

    if st.button("📋 Posizioni aperte", use_container_width=True):
        pos = bitget_positions(cfg_now)
        if pos: st.dataframe(pd.DataFrame(pos), use_container_width=True)
        else:   st.info("Nessuna posizione aperta.")

    st.divider()
    st.subheader("⚡ Ordine rapido")
    sig = res.get("signal","NEUTRAL")
    if sig in ("LONG","SHORT") and res.get("sl"):
        st.markdown(f"Segnale: **{sig}** su `{coin_sym}` — Entry `{res['entry_px']:.6g}`")
        with st.form("order_form"):
            oc1, oc2 = st.columns(2)
            with oc1:
                order_size = st.number_input("Dimensione (USDT)", 5.0, float(capital),
                                             float(min(capital*0.05,100)), step=5.0)
            with oc2:
                order_lev  = st.number_input("Leva", 1, 20, int(res.get("leverage",3)))
            auto_sl = st.number_input("Stop Loss ($)", value=round(float(res["sl"]),6))
            auto_tp = st.number_input("Take Profit ($)", value=round(float(res["tp1"]),6))
            if st.form_submit_button(f"🚀 Invia {sig}", type="primary"):
                if not cfg_now.get("api_key"):
                    st.error("Configura prima le chiavi API.")
                else:
                    r2 = bitget_place_order(cfg_now, coin_sym, sig,
                             order_size/max(res["entry_px"],1e-10),
                             auto_sl, auto_tp, order_lev)
                    if r2.get("code")=="00000":
                        st.success(f"✅ Ordine inviato! ID: {r2.get('data',{}).get('orderId','—')}")
                    else:
                        st.error(f"❌ {r2.get('msg',str(r2))}")
    else:
        st.info("Nessun segnale attivo. Genera un segnale LONG/SHORT dalla tab Analisi.")

    with st.expander("📖 Come creare le chiavi API su Bitget"):
        st.markdown("""
        1. Vai su **bitget.com → Profilo → API Management**
        2. Clicca **Create API** e scegli un nome (es. `JarvisPro`)
        3. Imposta una **Passphrase** e salvala
        4. Seleziona permessi: **Futures → Read + Order**
        5. Aggiungi il tuo IP in whitelist
        6. Completa la verifica 2FA
        7. Copia **API Key**, **Secret** e **Passphrase** nei campi qui sopra
        8. Lascia **DEMO attivo** finché non hai testato tutto
        """)

# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY TAB
# ─────────────────────────────────────────────────────────────────────────────

def render_accuracy_tab():
    st.subheader("🏆 Storico Segnali & Accuracy")
    logs = load_log()
    if not logs:
        st.info("Nessun segnale registrato ancora. Generane uno dalla tab Analisi."); return
    df_l = pd.DataFrame(logs)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Totale",  len(df_l))
    c2.metric("OPEN",    (df_l["result"]=="OPEN").sum())
    wins_n  = (df_l["result"]=="WIN").sum()
    losses_n= (df_l["result"]=="LOSS").sum()
    c3.metric("WIN",  wins_n)
    c4.metric("WR",   f"{wins_n/(wins_n+losses_n)*100:.0f}%" if wins_n+losses_n else "—")
    st.divider()
    for i, row in df_l.iloc[::-1].iterrows():
        with st.expander(f"#{row['id']} {row['symbol']} {row['tf']} — {row['signal']} — {row['result']}"):
            cc1,cc2,cc3,cc4 = st.columns(4)
            cc1.metric("Entry",  f"${row['entry']:.6g}")
            cc2.metric("SL",     f"${row['sl']:.6g}"  if row.get("sl")  else "—")
            cc3.metric("TP1",    f"${row['tp1']:.6g}" if row.get("tp1") else "—")
            cc4.metric("Score",  row["conf"])
            res_opt = st.selectbox("Risultato", ["OPEN","WIN","LOSS","BE"],
                key=f"res_{row['id']}",
                index=["OPEN","WIN","LOSS","BE"].index(row["result"]) \
                      if row["result"] in ["OPEN","WIN","LOSS","BE"] else 0)
            if res_opt != row["result"]:
                df_l.at[i,"result"] = res_opt
                save_log(df_l.to_dict("records"))
                st.success("Aggiornato!")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown("""<style>
    [data-testid="stMetric"]{background:#161b22;border-radius:8px;
        padding:10px 14px;border:1px solid #30363d;}
    </style>""", unsafe_allow_html=True)

    st.title("🧠 Jarvis Pro — Crypto AI Trading")
    st.caption("Binance · Motore V3 · ADX Gate · Kelly Sizing · Backtest Zero Bias")

    if "jarvis" not in st.session_state:
        st.session_state.jarvis = JarvisEngine()
    jarvis = st.session_state.jarvis

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Setup")

        # Caricamento coin — con feedback chiaro
        coins_placeholder = st.empty()
        with st.spinner("📡 Caricamento mercati..."):
            coins = get_top_crypto_pairs(200)
        coins_placeholder.caption(f"✅ {len(coins)} coppie USDT disponibili")

        srch = st.text_input("🔍 Cerca coin", "", placeholder="es. BTC, ETH, SOL...")
        filt = [c for c in coins if srch.upper() in c["coin"]] if srch else coins
        if not filt:
            st.warning("Nessuna coin trovata — prova un altro termine")
            filt = coins

        sel  = st.selectbox("💎 Seleziona Coin", [c["display"] for c in filt[:300]])
        coin = next((c for c in filt if c["display"] == sel), coins[0])
        symbol = coin["symbol"]
        st.info(f"📌 `{symbol}`")

        tf    = st.selectbox("⏱️ Timeframe", TIMEFRAMES_BIN,
                    index=TIMEFRAMES_BIN.index("1h"))
        limit = st.slider("📦 Candele", 100, 500, 300, step=50)

        st.divider()
        st.subheader("📊 Indicatori")
        ema_sel = st.multiselect("EMA",
            ["EMA_5","EMA_10","EMA_20","EMA_50","EMA_100","EMA_200"],
            default=["EMA_20","EMA_50","EMA_200"])
        ema_ps  = sorted([int(e.split("_")[1]) for e in ema_sel]) or [20, 50]

        overlays = st.multiselect("Overlay",
            ["BB","VWAP","SuperTrend","S/R"], default=["BB","SuperTrend","S/R"])
        oscillators = st.multiselect("Oscillatori",
            ["RSI","MACD","Stoch","ADX","CCI"], default=["RSI","MACD","ADX"])

        st.divider()
        st.subheader("💰 Risk Manager")
        capital = st.number_input("Capitale ($)", 100, 500000, 1000, step=500)
        risk_p  = st.slider("Rischio %", 0.5, 5.0, 1.0, step=0.5)
        rr_r    = st.slider("R:R", 1.0, 5.0, 2.0, step=0.5)

        st.divider()
        threshold = st.slider("🎯 Soglia segnale", 30, 80, 55,
            help="Score minimo per emettere LONG/SHORT. Abbassa se non vedi segnali.")
        h_chart = st.slider("📐 Altezza grafico", 600, 1200, 850, step=50)

        if st.button("🔄 Aggiorna dati", type="primary", use_container_width=True):
            st.cache_data.clear(); st.rerun()

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab_main, tab_bt, tab_api, tab_acc = st.tabs([
        "📊 Analisi & Segnale", "📈 Backtest", "🔑 API Bitget", "🏆 Accuracy"])

    # ── CARICAMENTO DATI ──────────────────────────────────────────────────────
    with st.spinner(f"📡 Download {symbol} {tf}..."):
        df_raw = get_ohlcv(symbol, tf, limit)

    if df_raw.empty:
        st.error(f"""
        ❌ Impossibile scaricare i dati per **{symbol}**.
        
        Possibili cause:
        - Connessione internet assente o limitata
        - Binance non raggiungibile dalla tua rete (VPN?)
        - Simbolo non valido
        
        👉 Prova ad aggiornare la pagina o selezionare un'altra coin.
        """)
        return

    if len(df_raw) < 50:
        st.warning(f"⚠️ Solo {len(df_raw)} candele disponibili — aumenta il limite o cambia timeframe.")

    df = add_all_indicators(df_raw, ema_ps)

    # HTF
    htf_map = {"1m":"5m","3m":"15m","5m":"15m","15m":"1h","30m":"1h",
               "1h":"4h","2h":"4h","4h":"1d","6h":"1d","8h":"1d",
               "12h":"1d","1d":"1w","3d":"1w","1w":"1M","1M":"1M"}
    htf_tf  = htf_map.get(tf,"4h")
    df_htf  = get_ohlcv(symbol, htf_tf, 100)
    htf_score = 50
    if not df_htf.empty and len(df_htf) >= 50:
        df_htf = add_all_indicators(df_htf, ema_ps)
        if "EMA_50" in df_htf.columns:
            htf_score = 65 if float(df_htf["close"].iloc[-1]) > float(df_htf["EMA_50"].iloc[-1]) else 35

    # Segnale
    res = jarvis.signal(df, ema_ps, threshold, htf_score)
    res["entry_px"] = float(df["close"].iloc[-1])

    # Aggiorna kNN
    if len(df) > 2:
        jarvis.update(df, 1 if df["close"].iloc[-1] > df["close"].iloc[-2] else -1)

    # Log segnale
    if res["signal"] != "NEUTRAL" and res.get("sl"):
        log_signal(coin["coin"], tf, res["signal"],
                   res["entry_px"], res["sl"], res["tp1"],
                   res["confidence"], res["leverage"])

    # ── TAB 1 ─────────────────────────────────────────────────────────────────
    with tab_main:
        px   = float(df["close"].iloc[-1])
        px_p = float(df["close"].iloc[-2]) if len(df) > 1 else px
        dpct = (px / px_p - 1) * 100 if px_p else 0
        px_f = f"${px:,.6f}" if px < 0.01 else f"${px:,.4f}" if px < 1 else f"${px:,.2f}"
        std_v = int(df["ST_dir"].iloc[-1]) if "ST_dir" in df.columns else 0

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("💰 Prezzo",      px_f, f"{dpct:+.2f}%")
        m2.metric("📉 RSI",         f"{df['RSI'].iloc[-1]:.1f}" if "RSI" in df.columns else "—")
        m3.metric("💪 ADX",         f"{res['adx']:.1f}")
        m4.metric("🌊 SuperTrend",  "🟢 BULL" if std_v==1 else "🔴 BEAR")
        m5.metric("📡 HTF Bias",    f"{htf_score}/100")
        m6.metric("📐 ATR%",        f"{res['atr_pct']:.2f}%")

        render_signal_card(res, capital, risk_p)

        # Debug info se NEUTRAL
        if res["signal"] == "NEUTRAL":
            with st.expander("🔍 Perché NEUTRAL? — Debug motore"):
                st.markdown(f"**ADX:** {res['adx']:.1f} (gate: >{ADX_TREND_GATE})")
                st.markdown(f"**ATR%:** {res['atr_pct']:.2f}% (gate: >{ATR_MIN_PCT}%)")
                st.markdown(f"**Long score:** {res.get('long_score',0)}/100 · **Short score:** {res.get('short_score',0)}/100")
                st.markdown(f"**Soglia attuale:** {threshold}/100")
                st.markdown("**Motivi SHORT:**")
                for r2 in res.get("reasons_short",[]): st.write(r2)
                st.info(f"💡 Prova ad abbassare la soglia (attuale {threshold}) a 45-50 o usa un timeframe con più trend (es. 4h, 1d).")

        with st.expander("🧠 Confluenze LONG vs SHORT"):
            dl, dr = st.columns(2)
            with dl:
                st.markdown(f"**🟢 LONG {res.get('long_score',0)}/100**")
                for r2 in res.get("reasons_long",[]): st.write(r2)
            with dr:
                st.markdown(f"**🔴 SHORT {res.get('short_score',0)}/100**")
                for r2 in res.get("reasons_short",[]): st.write(r2)

        fig = make_chart(df, ema_ps, overlays, oscillators, res, h_chart)
        st.plotly_chart(fig, use_container_width=True)

        # Calendario
        st.divider()
        st.subheader("📅 Calendario Economico")
        cc1, cc2, cc3 = st.columns([1,1,2])
        with cc1: sd = st.date_input("Da", datetime.now().date(), key="cd")
        with cc2: ed = st.date_input("A",  datetime.now().date()+timedelta(days=14), key="ca")
        with cc3: imp_f = st.multiselect("Impatto",["HIGH","MEDIUM","LOW"],
                           default=["HIGH","MEDIUM"], key="ci")

        with st.spinner("Caricamento calendario..."):
            ev_all = get_calendar()
        evs = sorted(
            [e for e in ev_all if e.get("date") and e["impact"] in imp_f
             and sd <= datetime.strptime(e["date"],"%Y-%m-%d").date() <= ed],
            key=lambda x: x["date"]+x.get("time",""))

        if evs:
            for e in evs:
                ic = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(e["impact"],"⚪")
                act = f" → **{e['actual']}**" if e.get("actual") else ""
                ce1, ce2 = st.columns([2,1])
                with ce1:
                    st.markdown(f"{ic} **{e['event']}**{act}")
                    st.caption(f"{e['country']} · {e['date']} {e.get('time','')}")
                with ce2:
                    st.caption(f"Prec: `{e['prev']}` · Prev: `{e['forecast']}`")
                st.divider()
        else:
            st.info("Nessun evento nel periodo selezionato.")

        with st.expander("📋 Dati recenti (tabella)"):
            cols_ = ["open","high","low","close","volume","RSI","MACD","ADX","ATR","CCI"]
            st.dataframe(df.tail(30)[[c for c in cols_ if c in df.columns]].round(6),
                use_container_width=True)

    # ── TAB 2 ─────────────────────────────────────────────────────────────────
    with tab_bt:
        st.subheader("📈 Backtest — Zero Look-Ahead Bias")
        st.caption(f"Warm-up 60 candele · ADX gate >{ADX_TREND_GATE} · ATR gate >{ATR_MIN_PCT}%")
        bt_cap = st.number_input("Capitale ($)", 100, 100000, 10000, step=1000, key="btcap")
        bt_thr = st.slider("Soglia", 30, 80, threshold, key="btthr")
        if st.button("▶️ Esegui Backtest", type="primary", use_container_width=True):
            with st.spinner("Calcolo..."):
                stats, equity = run_backtest(df, ema_ps, bt_thr, bt_cap)
            if stats:
                b1,b2,b3,b4 = st.columns(4)
                b1.metric("Win Rate",      f"{stats['wr']:.1f}%")
                b2.metric("Profit Factor", f"{stats['pf']:.2f}" if stats['pf']!=float('inf') else "∞")
                b3.metric("Max DD",        f"{stats['dd']:.1f}%")
                b4.metric("Return",        f"{stats['ret']:.1f}%", f"${stats['final']:,.0f}")
                st.caption(f"Trade: {stats['trades']} · W:{stats['wins']} L:{stats['losses']}")
                judge = "✅ Strategia buona" if stats["wr"]>55 and stats["pf"]>1.5 else \
                        "⚠️ Accettabile"     if stats["wr"]>45 else "❌ Rivedere parametri"
                st.markdown(f"**{judge}**")
                eq_fig = go.Figure(go.Scatter(y=equity, mode="lines",
                    line=dict(color="#00e676",width=2),
                    fill="tozeroy", fillcolor="rgba(0,230,118,0.07)"))
                eq_fig.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9"), height=300,
                    title="Equity Curve", margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(eq_fig, use_container_width=True)
            else:
                st.warning("Dati insufficienti (minimo 80 candele).")

    # ── TAB 3 ─────────────────────────────────────────────────────────────────
    with tab_api:
        render_bitget_settings(res, coin["coin"], capital)

    # ── TAB 4 ─────────────────────────────────────────────────────────────────
    with tab_acc:
        render_accuracy_tab()


main()
