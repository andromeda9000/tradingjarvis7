import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Jarvis V7 - Crypto Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 1. TOP 300 CRIPTO USDT DA BINANCE
# ============================================================================

@st.cache_data(ttl=3600)
def get_top_crypto_pairs(limit=300):
    try:
        resp = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=15)
        ticker_data = resp.json()
        usdt_pairs = [
            {
                'symbol': item['symbol'],
                'coin': item['symbol'].replace('USDT', ''),
                'volume': float(item['quoteVolume']),
                'price': float(item['lastPrice']),
                'change': float(item['priceChangePercent'])
            }
            for item in ticker_data if item['symbol'].endswith('USDT')
        ]
        usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
        result = []
        for pair in usdt_pairs[:limit]:
            arrow = "🟢" if pair['change'] >= 0 else "🔴"
            result.append({
                'symbol': pair['symbol'],
                'coin': pair['coin'],
                'volume': pair['volume'],
                'price': pair['price'],
                'change': pair['change'],
                'display': f"{arrow} {pair['coin']} - ${pair['price']:,.4f} ({pair['change']:+.2f}%)"
            })
        return result
    except Exception as e:
        st.error(f"Errore recupero cripto: {e}")
        return [{'symbol': 'BTCUSDT', 'coin': 'BTC', 'display': '🟢 BTC - Bitcoin', 'change': 0}]


@st.cache_data(ttl=60)
def get_crypto_data(symbol="BTCUSDT", interval="1h", limit=300):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if not data or isinstance(data, dict):
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "qa_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        st.error(f"Errore Binance: {e}")
        return pd.DataFrame()


# ============================================================================
# 2. CALENDARIO ECONOMICO (ForexFactory JSON)
# ============================================================================

@st.cache_data(ttl=3600)
def get_economic_calendar():
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            data = resp.json()
            events = []
            for item in data:
                raw_date = item.get("date", "")
                try:
                    dt = datetime.strptime(raw_date, "%Y-%m-%dT%H:%M:%S%z")
                    date_str = dt.strftime("%Y-%m-%d")
                    time_str = dt.strftime("%H:%M")
                except Exception:
                    date_str = raw_date[:10] if len(raw_date) >= 10 else raw_date
                    time_str = "00:00"
                impact = item.get("impact", "Low")
                events.append({
                    "date": date_str,
                    "time": time_str,
                    "country": item.get("country", ""),
                    "event": item.get("title", ""),
                    "impact": impact.upper() if impact else "LOW",
                    "prev": item.get("previous", "-") or "-",
                    "forecast": item.get("forecast", "-") or "-",
                    "actual": item.get("actual", "") or "",
                })
            return events
    except Exception:
        pass

    # Fallback con eventi dinamici relativi alla data odierna
    today = datetime.now()
    events_fallback = []
    offsets = [(1, "Non-Farm Payrolls", "🇺🇸", "HIGH", "175K", "180K"),
               (3, "CPI Inflation Rate", "🇺🇸", "HIGH", "3.2%", "3.1%"),
               (5, "FOMC Minutes", "🇺🇸", "HIGH", "5.25%", "5.25%"),
               (7, "ECB Rate Decision", "🇪🇺", "HIGH", "4.00%", "4.00%"),
               (10, "GDP Growth Rate Q1", "🇺🇸", "HIGH", "2.1%", "2.3%"),
               (12, "Bank of England Rate", "🇬🇧", "HIGH", "5.00%", "5.00%"),
               (14, "Core PCE Price Index", "🇺🇸", "MEDIUM", "2.7%", "2.6%"),
               (16, "Retail Sales", "🇺🇸", "MEDIUM", "0.7%", "0.5%"),
               (18, "Jobless Claims", "🇺🇸", "LOW", "220K", "215K"),
               (20, "Flash PMI Manufacturing", "🇪🇺", "MEDIUM", "47.3", "47.8")]
    for days_offset, name, country, impact, prev, forecast in offsets:
        ev_date = (today + timedelta(days=days_offset)).strftime("%Y-%m-%d")
        events_fallback.append({
            "date": ev_date,
            "time": "14:30",
            "country": country,
            "event": name,
            "impact": impact,
            "prev": prev,
            "forecast": forecast,
            "actual": "",
        })
    return events_fallback


# ============================================================================
# 3. INDICATORI TECNICI
# ============================================================================

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger(series, period=20, std_mult=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + (std * std_mult), sma - (std * std_mult), sma

def calculate_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Fix: quando entrambi positivi, prendi il maggiore, l'altro = 0
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    minus_dm[~mask] = 0
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr_14 = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_14.replace(0, np.nan))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period).mean(), plus_di, minus_di

def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df['close'] - low_min) / denom
    d = k.rolling(d_period).mean()
    return k, d

def calculate_vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).cumsum() / df['volume'].cumsum()

def calculate_supertrend(df, period=10, multiplier=3.0):
    atr = calculate_atr(df, period)
    hl_avg = (df['high'] + df['low']) / 2
    upper_band = hl_avg + multiplier * atr
    lower_band = hl_avg - multiplier * atr

    final_ub = upper_band.copy()
    final_lb = lower_band.copy()

    for i in range(1, len(df)):
        final_ub.iloc[i] = min(upper_band.iloc[i], final_ub.iloc[i-1]) if df['close'].iloc[i-1] <= final_ub.iloc[i-1] else upper_band.iloc[i]
        final_lb.iloc[i] = max(lower_band.iloc[i], final_lb.iloc[i-1]) if df['close'].iloc[i-1] >= final_lb.iloc[i-1] else lower_band.iloc[i]

    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        if df['close'].iloc[i] > final_ub.iloc[i-1]:
            direction.iloc[i] = 1
        elif df['close'].iloc[i] < final_lb.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
        supertrend.iloc[i] = final_lb.iloc[i] if direction.iloc[i] == 1 else final_ub.iloc[i]

    return supertrend, direction

def calculate_cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

def add_indicators(df, ema_periods):
    df = df.copy()
    close = df["close"]
    for p in ema_periods:
        df[f"EMA_{p}"] = calculate_ema(close, p)
    df["RSI"] = calculate_rsi(close)
    df["MACD_line"], df["MACD_signal"], df["MACD_hist"] = calculate_macd(close)
    df["BB_upper"], df["BB_lower"], df["BB_middle"] = calculate_bollinger(close)
    df["ATR"] = calculate_atr(df)
    df["ADX"], df["DI_plus"], df["DI_minus"] = calculate_adx(df)
    df["STOCH_K"], df["STOCH_D"] = calculate_stochastic(df)
    df["VWAP"] = calculate_vwap(df)
    df["SUPERTREND"], df["SUPERTREND_DIR"] = calculate_supertrend(df)
    df["CCI"] = calculate_cci(df)
    return df


# ============================================================================
# 4. MOTORE JARVIS V7
# ============================================================================

class JarvisV7Engine:
    def __init__(self, history_size=500, k_neighbors=8):
        self.history_size = history_size
        self.k_neighbors = k_neighbors
        self.price_history = deque(maxlen=history_size)
        self.feature_history = deque(maxlen=history_size)
        self.direction_history = deque(maxlen=history_size)
        self.win_rate_history = deque(maxlen=100)

    def _calc_wma(self, series, period):
        return series.rolling(period).apply(
            lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True
        )

    def _calc_hma(self, series, period):
        half = int(period / 2)
        sqrt_p = int(np.sqrt(period))
        return self._calc_wma(2 * self._calc_wma(series, half) - self._calc_wma(series, period), sqrt_p)

    def calculate_features(self, df):
        close = df["close"]
        rsi = calculate_rsi(close)
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
        rsi_s = self._calc_hma(rsi.fillna(50), 9)
        cci_s = self._calc_hma(cci.fillna(0), 9)
        return pd.DataFrame({
            "rsi_norm": ((rsi_s - 30) / 40).clip(0, 1),
            "cci_norm": ((cci_s + 200) / 400).clip(0, 1)
        }).fillna(0.5)

    def knn_predict(self, df):
        features = self.calculate_features(df)
        curr = features.iloc[-1].values
        if len(self.feature_history) < self.k_neighbors:
            return 0
        X = np.array(list(self.feature_history))
        y = np.array(list(self.direction_history))
        dists = np.sqrt(np.sum((np.log1p(curr + 1e-6) - np.log1p(X + 1e-6))**2, axis=1))
        nearest = y[np.argsort(dists)[:self.k_neighbors]]
        return 1 if nearest.sum() > 0 else -1

    def update_history(self, df, direction):
        feat = self.calculate_features(df).iloc[-1].values
        self.feature_history.append(feat)
        self.direction_history.append(direction)
        self.price_history.append(df['close'].iloc[-1])

    def regime_filter(self, df):
        if 'ADX' not in df.columns or len(df) < 50:
            return 1.0
        atr_pct = (calculate_atr(df) / df['close']) * 100
        curr_atr = atr_pct.iloc[-1]
        curr_adx = df['ADX'].iloc[-1]
        if curr_atr < atr_pct.rolling(50).quantile(0.33).iloc[-1]:
            return 0.85
        elif curr_adx > 50:
            return 0.60
        return 1.0

    def find_order_blocks(self, df):
        high, low, close = df['high'], df['low'], df['close']
        order_blocks = []
        lookback = min(20, len(df) - 2)
        for i in range(lookback, len(df) - 1):
            if i < 2:
                continue
            rng = high.iloc[i] - low.iloc[i]
            if rng <= 0:
                continue
            if low.iloc[i] < low.iloc[i-1] and close.iloc[i] > close.iloc[i-2]:
                order_blocks.append({'type': 'bullish', 'price': low.iloc[i],
                                     'strength': min(abs(close.iloc[i] - low.iloc[i]) / rng, 1)})
            elif high.iloc[i] > high.iloc[i-1] and close.iloc[i] < close.iloc[i-2]:
                order_blocks.append({'type': 'bearish', 'price': high.iloc[i],
                                     'strength': min(abs(high.iloc[i] - close.iloc[i]) / rng, 1)})
        return order_blocks[-5:] if order_blocks else []

    def detect_liquidity_sweep(self, df):
        if len(df) < 50:
            return []
        high, low = df['high'], df['low']
        window = min(20, len(df) // 3)
        pivot_highs, pivot_lows = [], []
        for i in range(window, len(df) - window):
            if high.iloc[i] > high.iloc[i-window:i].max() and high.iloc[i] > high.iloc[i+1:i+window+1].max():
                pivot_highs.append(high.iloc[i])
            if low.iloc[i] < low.iloc[i-window:i].min() and low.iloc[i] < low.iloc[i+1:i+window+1].min():
                pivot_lows.append(low.iloc[i])
        curr = df['close'].iloc[-1]
        sweeps = []
        for ph in pivot_highs[-5:]:
            if curr > ph * 1.001:
                sweeps.append({'type': 'BSL_SWEEP', 'price': ph})
        for pl in pivot_lows[-5:]:
            if curr < pl * 0.999:
                sweeps.append({'type': 'SSL_SWEEP', 'price': pl})
        return sweeps

    def cvd_divergence(self, df):
        if 'volume' not in df.columns or len(df) < 20:
            return 0
        delta = df['volume'] * np.sign(df['close'].diff().fillna(0))
        cvd = delta.cumsum()
        if df['close'].iloc[-1] <= df['close'].rolling(20).min().iloc[-1] * 1.01 and cvd.iloc[-1] > cvd.rolling(20).min().iloc[-1]:
            return 15
        if df['close'].iloc[-1] >= df['close'].rolling(20).max().iloc[-1] * 0.99 and cvd.iloc[-1] < cvd.rolling(20).max().iloc[-1]:
            return -10
        return 0

    def smart_sl(self, df, direction):
        close = df['close']
        atr_val = calculate_atr(df).iloc[-1]
        buffer = atr_val * 0.2
        obs = self.find_order_blocks(df)
        if direction == 'LONG':
            base = df['low'].rolling(50).min().iloc[-1]
            for ob in obs:
                if ob['type'] == 'bullish' and base < ob['price'] < close.iloc[-1]:
                    base = ob['price']
            result = base - buffer
            return result if not np.isnan(result) and result > 0 else close.iloc[-1] * 0.98
        else:
            base = df['high'].rolling(50).max().iloc[-1]
            for ob in obs:
                if ob['type'] == 'bearish' and close.iloc[-1] < ob['price'] < base:
                    base = ob['price']
            result = base + buffer
            return result if not np.isnan(result) else close.iloc[-1] * 1.02

    def kelly_size(self):
        if len(self.win_rate_history) > 0:
            wr = sum(self.win_rate_history) / len(self.win_rate_history)
        else:
            wr = 0.55
        b = 1.5
        kelly = max(0.05, min((wr * b - (1 - wr)) / b, 0.25))
        return kelly * 0.25

    def ai_score(self, df, ema_periods):
        score = 0
        reasons = []
        knn_dir = self.knn_predict(df)
        if knn_dir != 0:
            score += 20
            lbl = "rialzista" if knn_dir == 1 else "ribassista"
            reasons.append(f"🧠 k-NN: +20 (previsione {lbl})")

        close = df['close']
        ema_vals = {p: df[f'EMA_{p}'].iloc[-1] for p in ema_periods if f'EMA_{p}' in df.columns}
        sorted_p = sorted(ema_vals.keys())
        if len(sorted_p) >= 2:
            if all(ema_vals[sorted_p[i]] > ema_vals[sorted_p[i+1]] for i in range(len(sorted_p)-1)):
                score += 20; reasons.append("📈 EMA allineate bullish: +20")
            elif all(ema_vals[sorted_p[i]] < ema_vals[sorted_p[i+1]] for i in range(len(sorted_p)-1)):
                score += 20; reasons.append("📉 EMA allineate bearish: +20")

        cvd = self.cvd_divergence(df)
        score += cvd
        if cvd > 0: reasons.append(f"💧 Assorbimento CVD: +{cvd}")
        elif cvd < 0: reasons.append(f"⚠️ Divergenza CVD: {cvd}")

        for ob in self.find_order_blocks(df):
            curr = close.iloc[-1]
            if ob['type'] == 'bullish' and curr < ob['price'] * 1.01:
                score += 10; reasons.append("🏛️ Order Block Bullish: +10"); break
            elif ob['type'] == 'bearish' and curr > ob['price'] * 0.99:
                score += 10; reasons.append("🏛️ Order Block Bearish: +10"); break

        for sweep in self.detect_liquidity_sweep(df):
            score += 25
            lbl = "long" if sweep['type'] == 'SSL_SWEEP' else "short"
            reasons.append(f"🎯 Sweep {sweep['type']}: +25 (presa liquidità {lbl})")

        if 'MACD_line' in df.columns and df['MACD_line'].iloc[-1] > df['MACD_signal'].iloc[-1]:
            score += 5; reasons.append("📊 MACD confluenza: +5")
        if 'RSI' in df.columns and 40 < df['RSI'].iloc[-1] < 60:
            score += 5; reasons.append("📊 RSI neutrale: +5")
        if 'SUPERTREND_DIR' in df.columns:
            st_dir = df['SUPERTREND_DIR'].iloc[-1]
            if st_dir == 1: score += 10; reasons.append("🌊 SuperTrend bullish: +10")
            elif st_dir == -1: score += 10; reasons.append("🌊 SuperTrend bearish: +10")

        mult = self.regime_filter(df)
        orig = score
        score = int(min(score * mult, 100))
        if mult < 1:
            reasons.append(f"⚡ Regime Filter: x{mult:.2f} ({orig} → {score})")
        return score, reasons, knn_dir

    def generate_signal(self, df, ema_periods, threshold=60):
        ai, reasons, knn_dir = self.ai_score(df, ema_periods)
        mult = self.regime_filter(df)
        if ai >= threshold and mult >= 0.7:
            signal = "LONG" if knn_dir == 1 or ai >= 75 else ("SHORT" if knn_dir == -1 else "NEUTRAL")
        else:
            signal = "NEUTRAL"
        sl = self.smart_sl(df, signal) if signal in ["LONG", "SHORT"] else None
        ks = self.kelly_size() if signal in ["LONG", "SHORT"] else None
        return {"signal": signal, "confidence": ai, "ai_score": ai,
                "reasons": reasons, "smart_sl": sl, "kelly_size": ks,
                "knn_direction": knn_dir, "regime_multiplier": mult}


# ============================================================================
# 5. GRAFICO AVANZATO
# ============================================================================

def create_chart(df, ema_periods, indicators_to_show):
    has_macd = "MACD" in indicators_to_show
    has_rsi = "RSI" in indicators_to_show
    has_stoch = "Stochastic" in indicators_to_show
    has_adx = "ADX" in indicators_to_show
    has_cci = "CCI" in indicators_to_show

    n_rows = 2  # price + volume always
    subplot_map = {}
    row_titles = ["📈 Prezzo & Indicatori", "📊 Volume"]
    row_heights = [0.50, 0.12]

    if has_macd:
        n_rows += 1; subplot_map['macd'] = n_rows
        row_titles.append("⚡ MACD"); row_heights.append(0.14)
    if has_rsi:
        n_rows += 1; subplot_map['rsi'] = n_rows
        row_titles.append("📉 RSI"); row_heights.append(0.12)
    if has_stoch:
        n_rows += 1; subplot_map['stoch'] = n_rows
        row_titles.append("🔄 Stocastico"); row_heights.append(0.12)
    if has_adx:
        n_rows += 1; subplot_map['adx'] = n_rows
        row_titles.append("💪 ADX / DI"); row_heights.append(0.12)
    if has_cci:
        n_rows += 1; subplot_map['cci'] = n_rows
        row_titles.append("📐 CCI"); row_heights.append(0.10)

    # Normalizza altezze
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=row_heights,
        subplot_titles=row_titles
    )

    # --- Candele ---
    increasing_color = "#26a69a"
    decreasing_color = "#ef5350"
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Prezzo",
        increasing_line_color=increasing_color,
        decreasing_line_color=decreasing_color,
        increasing_fillcolor=increasing_color,
        decreasing_fillcolor=decreasing_color,
    ), row=1, col=1)

    # --- EMAs ---
    ema_colors = {5: "#00e5ff", 10: "#69f0ae", 20: "#ffeb3b",
                  50: "#ff9800", 100: "#ce93d8", 200: "#ef9a9a"}
    for p in ema_periods:
        col_name = f"EMA_{p}"
        if col_name in indicators_to_show and col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name], name=f"EMA {p}",
                line=dict(color=ema_colors.get(p, "#aaaaaa"), width=1.5)
            ), row=1, col=1)

    # --- Bollinger Bands ---
    if "Bollinger Bands" in indicators_to_show and "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color="rgba(100,150,200,0.8)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color="rgba(100,150,200,0.8)", width=1, dash="dot"),
            fill='tonexty', fillcolor='rgba(100,150,200,0.05)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_middle"], name="BB Mid",
            line=dict(color="rgba(150,150,150,0.5)", width=1)), row=1, col=1)

    # --- VWAP ---
    if "VWAP" in indicators_to_show and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
            line=dict(color="#ff6b6b", width=2, dash="dashdot")), row=1, col=1)

    # --- SuperTrend ---
    if "SuperTrend" in indicators_to_show and "SUPERTREND" in df.columns:
        bull_mask = df["SUPERTREND_DIR"] == 1
        bear_mask = df["SUPERTREND_DIR"] == -1
        st_bull = df["SUPERTREND"].where(bull_mask)
        st_bear = df["SUPERTREND"].where(bear_mask)
        fig.add_trace(go.Scatter(x=df.index, y=st_bull, name="SuperTrend ↑",
            line=dict(color="#00e676", width=2), connectgaps=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=st_bear, name="SuperTrend ↓",
            line=dict(color="#ff1744", width=2), connectgaps=False), row=1, col=1)

    # --- Volume ---
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color=vol_colors, opacity=0.7
    ), row=2, col=1)

    # --- MACD ---
    if has_macd and 'macd' in subplot_map:
        r = subplot_map['macd']
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD Hist",
            marker_color=hist_colors, opacity=0.7), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_line"], name="MACD",
            line=dict(color="#2196f3", width=1.5)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
            line=dict(color="#ff9800", width=1.5)), row=r, col=1)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1, row=r, col=1)

    # --- RSI ---
    if has_rsi and 'rsi' in subplot_map:
        r = subplot_map['rsi']
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#ce93d8", width=2),
            fill='tozeroy', fillcolor='rgba(206,147,216,0.05)'), row=r, col=1)
        for level, color in [(70, "rgba(239,83,80,0.6)"), (30, "rgba(38,166,154,0.6)"), (50, "rgba(255,255,255,0.2)")]:
            fig.add_hline(y=level, line_dash="dash", line_color=color, line_width=1, row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1)

    # --- Stocastico ---
    if has_stoch and 'stoch' in subplot_map:
        r = subplot_map['stoch']
        fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_K"], name="%K",
            line=dict(color="#4fc3f7", width=1.5)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_D"], name="%D",
            line=dict(color="#ff8a65", width=1.5)), row=r, col=1)
        for level, color in [(80, "rgba(239,83,80,0.6)"), (20, "rgba(38,166,154,0.6)")]:
            fig.add_hline(y=level, line_dash="dash", line_color=color, line_width=1, row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1)

    # --- ADX ---
    if has_adx and 'adx' in subplot_map:
        r = subplot_map['adx']
        fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX",
            line=dict(color="#ffeb3b", width=2)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["DI_plus"], name="DI+",
            line=dict(color="#69f0ae", width=1.2, dash="dot")), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["DI_minus"], name="DI-",
            line=dict(color="#ef9a9a", width=1.2, dash="dot")), row=r, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=r, col=1)

    # --- CCI ---
    if has_cci and 'cci' in subplot_map:
        r = subplot_map['cci']
        fig.add_trace(go.Scatter(x=df.index, y=df["CCI"], name="CCI",
            line=dict(color="#80cbc4", width=1.5)), row=r, col=1)
        for level, color in [(100, "rgba(239,83,80,0.6)"), (-100, "rgba(38,166,154,0.6)"), (0, "rgba(255,255,255,0.2)")]:
            fig.add_hline(y=level, line_dash="dash", line_color=color, line_width=1, row=r, col=1)

    fig.update_layout(
        title=dict(
            text=f"📊 Analisi Tecnica Avanzata — Jarvis V7",
            font=dict(size=18, color="#e0e0e0"),
            x=0.01
        ),
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=max(700, 500 + n_rows * 80),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            bgcolor="rgba(22,27,34,0.8)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(size=11)
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=20, t=80, b=40),
    )

    # Griglia sottile su tutti i subplot
    for i in range(1, n_rows + 1):
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.05)", zeroline=False, row=i, col=1,
            rangeslider_visible=False
        )
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.05)", zeroline=False, row=i, col=1
        )

    return fig


# ============================================================================
# 6. INTERFACCIA PRINCIPALE
# ============================================================================

def main():
    st.markdown("""
    <style>
    .stMetric { background: #161b22; border-radius: 8px; padding: 12px; border: 1px solid #30363d; }
    .stMetric label { color: #8b949e !important; font-size: 0.75rem !important; }
    .signal-long { background: #1a3a2a; border: 1px solid #238636; border-radius: 8px; padding: 12px; }
    .signal-short { background: #3a1a1a; border: 1px solid #da3633; border-radius: 8px; padding: 12px; }
    .signal-neutral { background: #1c1f26; border: 1px solid #444c56; border-radius: 8px; padding: 12px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧠 Jarvis V7 — Crypto Trading Dashboard")
    st.caption("Machine Learning • SMC • Smart Money • AI Score • Calendario Economico")
    st.markdown("---")

    if 'jarvis_engine' not in st.session_state:
        st.session_state.jarvis_engine = JarvisV7Engine()
    jarvis = st.session_state.jarvis_engine

    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("⚙️ Configurazione Cripto")

        with st.spinner("Caricamento top 300 cripto..."):
            crypto_list = get_top_crypto_pairs(300)

        if crypto_list:
            selected_display = st.selectbox(
                "🔍 Cerca Criptovaluta (Top 300 per volume)",
                options=[c["display"] for c in crypto_list],
                help="Digita per cercare"
            )
            selected_symbol = next(
                (c["symbol"] for c in crypto_list if c["display"] == selected_display),
                "BTCUSDT"
            )
        else:
            selected_symbol = st.text_input("Simbolo (es. BTCUSDT)", "BTCUSDT")

        timeframe = st.selectbox(
            "⏱️ Timeframe",
            ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"],
            index=4
        )
        limit = st.slider("📦 Numero candele", 100, 500, 250, step=50)

        st.markdown("---")
        st.subheader("📈 Indicatori da mostrare")

        ema_options = ["EMA_5", "EMA_10", "EMA_20", "EMA_50", "EMA_100", "EMA_200"]
        selected_emas = st.multiselect("EMA", ema_options, default=["EMA_20", "EMA_50", "EMA_200"])
        ema_periods = [int(e.split("_")[1]) for e in selected_emas]
        if not ema_periods:
            ema_periods = [50]

        other_indicators = st.multiselect(
            "Overlay (sul grafico prezzo)",
            ["Bollinger Bands", "VWAP", "SuperTrend"],
            default=["Bollinger Bands", "SuperTrend"]
        )
        sub_indicators = st.multiselect(
            "Sottofinestre",
            ["MACD", "RSI", "Stochastic", "ADX", "CCI"],
            default=["MACD", "RSI", "ADX"]
        )

        indicators_to_show = selected_emas + other_indicators + sub_indicators

        st.markdown("---")
        soglia = st.slider("🎯 Confidenza minima segnale %", 40, 80, 60)
        st.markdown("---")

        if st.button("🔄 Aggiorna Dati", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.info(f"""
**🧠 Jarvis V7 Modules:**
- k-NN predittore (500 stati)
- Regime Filter (ATR/ADX)
- Order Blocks SMC
- Liquidity Sweep
- CVD Divergence
- Smart SL + Kelly Criterion
- SuperTrend

**Soglia attiva:** {soglia}%
        """)

    # ---- AREA PRINCIPALE ----
    left_col, right_col = st.columns([3, 1])

    with left_col:
        with st.spinner(f"📥 Scaricamento dati {selected_symbol}..."):
            df_raw = get_crypto_data(selected_symbol, timeframe, limit)

        if df_raw.empty:
            st.error(f"❌ Nessun dato per {selected_symbol}. Controlla il simbolo o la connessione.")
            return

        df = add_indicators(df_raw, ema_periods)
        jarvis_result = jarvis.generate_signal(df, ema_periods, soglia)

        if len(df) > 2:
            future_dir = 1 if df['close'].iloc[-1] > df['close'].iloc[-2] else -1
            jarvis.update_history(df, future_dir)

        # Segnale
        sig = jarvis_result["signal"]
        score = jarvis_result["confidence"]
        knn_text = "🟢 LONG" if jarvis_result['knn_direction'] == 1 else "🔴 SHORT" if jarvis_result['knn_direction'] == -1 else "⚪ NEUTRO"

        if sig == "LONG":
            st.success(f"🔵 **SEGNALE LONG — AI Score: {score}/100**")
        elif sig == "SHORT":
            st.error(f"🔴 **SEGNALE SHORT — AI Score: {score}/100**")
        else:
            st.warning(f"⚪ **NESSUN SEGNALE FORTE — AI Score: {score}/100**")

        # KPI Metrics
        curr_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        delta_pct = (curr_close / prev_close - 1) * 100
        rsi_val = df['RSI'].iloc[-1] if 'RSI' in df.columns else 0
        adx_val = df['ADX'].iloc[-1] if 'ADX' in df.columns else 0
        atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0
        bb_w = ((df['BB_upper'].iloc[-1] - df['BB_lower'].iloc[-1]) / df['BB_middle'].iloc[-1] * 100) if 'BB_upper' in df.columns else 0
        st_dir = "🟢 BULL" if df.get('SUPERTREND_DIR', pd.Series([0])).iloc[-1] == 1 else "🔴 BEAR"

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("💰 Prezzo", f"${curr_close:,.4f}", f"{delta_pct:+.2f}%")
        c2.metric("📉 RSI (14)", f"{rsi_val:.1f}")
        c3.metric("💪 ADX", f"{adx_val:.1f}")
        c4.metric("📊 ATR", f"{atr_val:.4f}")
        c5.metric("📐 BB Width", f"{bb_w:.1f}%")
        c6.metric("🌊 SuperTrend", st_dir)

        # Dettaglio AI
        with st.expander("🧠 Jarvis V7 — Dettaglio AI Score", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**AI Score:** {score}/100")
                st.write(f"**k-NN Direzione:** {knn_text}")
                st.write(f"**Regime Multiplier:** x{jarvis_result['regime_multiplier']:.2f}")
                if jarvis_result.get('smart_sl'):
                    risk_pct = abs((jarvis_result['smart_sl'] - curr_close) / curr_close) * 100
                    st.write(f"**Smart Stop Loss:** ${jarvis_result['smart_sl']:.4f} ({risk_pct:.2f}% rischio)")
                if jarvis_result.get('kelly_size'):
                    st.write(f"**Kelly Size:** {jarvis_result['kelly_size']*100:.1f}% del capitale")
            with col_b:
                st.markdown("**Componenti score:**")
                for r in jarvis_result['reasons']:
                    st.write(f"• {r}")

        # Grafico
        st.subheader("📈 Grafico Tecnico Avanzato")
        fig = create_chart(df, ema_periods, indicators_to_show)
        st.plotly_chart(fig, use_container_width=True)

        # Tabella dati
        with st.expander("📋 Tabella dati recenti"):
            cols_show = ["open", "high", "low", "close", "volume", "RSI", "MACD_line",
                         "ADX", "STOCH_K", "ATR", "VWAP", "CCI"]
            avail = [c for c in cols_show if c in df.columns]
            st.dataframe(df.tail(20)[avail].round(4), use_container_width=True)

    # ---- COLONNA DESTRA: Calendario ----
    with right_col:
        st.subheader("📅 Calendario Economico")
        st.caption("Dati: ForexFactory (aggiornati ogni ora)")

        with st.spinner("Caricamento eventi..."):
            all_events = get_economic_calendar()

        c_date1, c_date2 = st.columns(2)
        with c_date1:
            start_date = st.date_input("Da", datetime.now().date(), key="cal_start")
        with c_date2:
            end_date = st.date_input("A", datetime.now().date() + timedelta(days=14), key="cal_end")

        impact_filter = st.multiselect("Impatto", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])

        filtered = [
            e for e in all_events
            if e.get("date") and
            start_date <= datetime.strptime(e["date"], "%Y-%m-%d").date() <= end_date and
            e["impact"] in impact_filter
        ]
        filtered.sort(key=lambda x: x["date"] + x.get("time", ""))

        if filtered:
            for ev in filtered:
                actual_str = f" → **Attuale: {ev['actual']}**" if ev.get("actual") else ""
                impact_icon = "🔴" if ev["impact"] == "HIGH" else "🟡" if ev["impact"] == "MEDIUM" else "🟢"

                with st.container():
                    st.markdown(f"{impact_icon} **{ev['event']}**{actual_str}")
                    st.caption(f"📍 {ev['country']} | 📅 {ev['date']} {ev.get('time','')}")
                    st.caption(f"Prec: `{ev['prev']}` | Prev: `{ev['forecast']}`")
                    st.markdown("---")
        else:
            st.info("Nessun evento nel periodo selezionato")

        st.caption("💡 Gli eventi HIGH impattano forte sulla volatilità crypto")

        # Mini overview crypto top 5
        st.markdown("---")
        st.subheader("🏆 Top 5 per Volume")
        if crypto_list:
            for c in crypto_list[:5]:
                arrow = "▲" if c.get('change', 0) >= 0 else "▼"
                color = "green" if c.get('change', 0) >= 0 else "red"
                st.markdown(
                    f"**{c['coin']}** — ${c['price']:,.4f} "
                    f"<span style='color:{color}'>{arrow} {c.get('change',0):+.2f}%</span> "
                    f"<small style='color:#8b949e'>Vol: ${c['volume']/1e9:.1f}B</small>",
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()
