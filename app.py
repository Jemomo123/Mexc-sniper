# Version 19 - Pure Trader Logic (Scalper First, Swing Second)
import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import time

st.set_page_config(page_title="Scalper Pro Scanner", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {padding: 0.5rem !important; max-width: 100% !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stButton>button {width: 100%; border-radius: 12px; height: 3.5em; font-weight: bold; font-size: 1rem; margin: 0.5rem 0;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem; font-weight: bold;}
    @media (max-width: 768px) {
        .row-widget.stHorizontal {flex-direction: column !important;}
        div[data-testid="column"] {width: 100% !important; margin-bottom: 1rem;}
        .stButton>button {font-size: 1.1rem; height: 4em;}
    }
</style>
""", unsafe_allow_html=True)

if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = True

if st.session_state.auto_refresh_enabled:
    st_autorefresh(interval=60000, key="datarefresh")

@st.cache_resource
def init_exchange(exchange_name='mexc'):
    try:
        if exchange_name == 'mexc':
            exchange = ccxt.mexc({'enableRateLimit': True, 'options': {'defaultType': 'spot'}, 'timeout': 30000})
        elif exchange_name == 'gateio':
            exchange = ccxt.gateio({'enableRateLimit': True, 'options': {'defaultType': 'spot'}, 'timeout': 30000})
        else:
            exchange = ccxt.mexc({'enableRateLimit': True, 'options': {'defaultType': 'spot'}, 'timeout': 30000})
        return exchange, exchange_name
    except Exception as e:
        return None, None

def fetch_ohlcv(exchange, symbol, timeframe, limit=200):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except:
        return None

def calculate_indicators(df):
    if df is None or len(df) < 100:
        return None
    try:
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_100'] = ta.sma(df['close'], length=100)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['range'] = df['high'] - df['low']
        df['avg_range'] = df['range'].rolling(20).mean()
        df['body'] = abs(df['close'] - df['open'])
        return df
    except:
        return None

def check_liquidity(exchange, symbol):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
        threshold = mid_price * 0.005
        bid_depth = sum([bid[1] for bid in orderbook['bids'][:10] if bid[0] >= mid_price - threshold])
        ask_depth = sum([ask[1] for ask in orderbook['asks'][:10] if ask[0] <= mid_price + threshold])
        hole_above = ask_depth < bid_depth * 0.3
        hole_below = bid_depth < ask_depth * 0.3
        return {'hole_above': hole_above, 'hole_below': hole_below}
    except:
        return {'hole_above': False, 'hole_below': False}

def detect_signals_trader_logic(df, df_higher, symbol, timeframe, liquidity_data):
    if df is None or len(df) < 100:
        return []
    
    signals = []
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    if pd.isna(latest['sma_20']) or pd.isna(latest['sma_100']) or pd.isna(latest['rsi']):
        return []
    
    # Check higher timeframe bias (15m for scalps, 1h for swings)
    higher_bias_bullish = True
    higher_bias_bearish = True
    if df_higher is not None and len(df_higher) >= 100:
        latest_higher = df_higher.iloc[-1]
        if not pd.isna(latest_higher['sma_100']):
            higher_bias_bullish = latest_higher['close'] > latest_higher['sma_100']
            higher_bias_bearish = latest_higher['close'] < latest_higher['sma_100']
    
    # Current timeframe conditions
    price_above_100 = latest['close'] > latest['sma_100']
    price_below_100 = latest['close'] < latest['sma_100']
    sma20_above_100 = latest['sma_20'] > latest['sma_100']
    sma20_below_100 = latest['sma_20'] < latest['sma_100']
    
    # SMA 20 momentum
    sma20_crossing_up = prev['sma_20'] <= prev['sma_100'] and latest['sma_20'] > latest['sma_100']
    sma20_crossing_down = prev['sma_20'] >= prev['sma_100'] and latest['sma_20'] < latest['sma_100']
    sma20_pullback_hold = sma20_above_100 and latest['close'] > latest['sma_20']
    sma20_rejection = sma20_below_100 and latest['close'] < latest['sma_20']
    
    # RSI zones
    is_scalp = timeframe in ['3m', '5m']
    is_swing = timeframe in ['15m', '1h', '4h']
    
    # Elephant/Tail bars
    elephant_bar = False
    if not pd.isna(latest['avg_range']):
        elephant_bar = latest['range'] > latest['avg_range'] * 2.0 and latest['body'] > latest['range'] * 0.65
    is_bullish_bar = latest['close'] > latest['open']
    is_bearish_bar = latest['close'] < latest['open']
    
    # Chop detection
    sma_diff = abs(latest['sma_20'] - latest['sma_100']) / latest['sma_100'] * 100
    is_choppy = sma_diff < 0.3
    
    # SCALP LOGIC (3m, 5m)
    if is_scalp:
        # LONG SCALP
        if higher_bias_bullish and price_above_100 and (sma20_crossing_up or sma20_pullback_hold):
            if 45 <= latest['rsi'] <= 60:
                # Calculate confidence
                conf_score = 0
                entry_reasons = []
                
                if higher_bias_bullish:
                    conf_score += 1
                    entry_reasons.append("15m Bias Bullish")
                if sma20_crossing_up:
                    conf_score += 2
                    entry_reasons.append("SMA 20 Cross UP")
                elif sma20_pullback_hold:
                    conf_score += 1
                    entry_reasons.append("SMA 20 Pullback Hold")
                if 50 <= latest['rsi'] <= 55:
                    conf_score += 2
                    entry_reasons.append("RSI Perfect Zone")
                elif 45 <= latest['rsi'] <= 60:
                    conf_score += 1
                    entry_reasons.append("RSI Healthy")
                if elephant_bar and is_bullish_bar:
                    conf_score += 2
                    entry_reasons.append("Elephant Bar Present")
                if liquidity_data['hole_above']:
                    conf_score += 1
                    entry_reasons.append("Vacuum Above (Easy Move)")
                
                # Determine confidence level
                if conf_score >= 6:
                    confidence = "🟢 HIGH"
                elif conf_score >= 4:
                    confidence = "🟡 MEDIUM"
                elif conf_score >= 2:
                    confidence = "🟠 CAUTION"
                else:
                    confidence = "⚪ WAIT"
                
                vacuum = " + VACUUM" if liquidity_data['hole_above'] else ""
                signals.append({
                    'symbol': symbol, 'timeframe': timeframe, 'exchange': '',
                    'strategy': f"⚡ LONG SCALP{vacuum}",
                    'direction': 'UP', 'price': latest['close'],
                    'confidence': confidence,
                    'conf_score': conf_score,
                    'entry_reason': " | ".join(entry_reasons),
                    'conditions': {
                        'Mode': 'SCALP',
                        'Bias (15m)': 'Bullish',
                        'SMA 20': 'Momentum UP',
                        'RSI': f"{latest['rsi']:.0f}",
                        'Momentum': 'Elephant Bar' if elephant_bar else 'Normal'
                    },
                    'warning': 'Thin Liquidity Below' if liquidity_data['hole_below'] else '',
                    'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # SHORT SCALP
        if higher_bias_bearish and price_below_100 and (sma20_crossing_down or sma20_rejection):
            if 40 <= latest['rsi'] <= 55:
                conf_score = 0
                entry_reasons = []
                
                if higher_bias_bearish:
                    conf_score += 1
                    entry_reasons.append("15m Bias Bearish")
                if sma20_crossing_down:
                    conf_score += 2
                    entry_reasons.append("SMA 20 Cross DOWN")
                elif sma20_rejection:
                    conf_score += 1
                    entry_reasons.append("SMA 20 Rejection")
                if 45 <= latest['rsi'] <= 50:
                    conf_score += 2
                    entry_reasons.append("RSI Perfect Zone")
                elif 40 <= latest['rsi'] <= 55:
                    conf_score += 1
                    entry_reasons.append("RSI Healthy")
                if elephant_bar and is_bearish_bar:
                    conf_score += 2
                    entry_reasons.append("Elephant Bar Present")
                if liquidity_data['hole_below']:
                    conf_score += 1
                    entry_reasons.append("Vacuum Below (Easy Move)")
                
                if conf_score >= 6:
                    confidence = "🟢 HIGH"
                elif conf_score >= 4:
                    confidence = "🟡 MEDIUM"
                elif conf_score >= 2:
                    confidence = "🟠 CAUTION"
                else:
                    confidence = "⚪ WAIT"
                
                vacuum = " + VACUUM" if liquidity_data['hole_below'] else ""
                signals.append({
                    'symbol': symbol, 'timeframe': timeframe, 'exchange': '',
                    'strategy': f"⚡ SHORT SCALP{vacuum}",
                    'direction': 'DOWN', 'price': latest['close'],
                    'confidence': confidence,
                    'conf_score': conf_score,
                    'entry_reason': " | ".join(entry_reasons),
                    'conditions': {
                        'Mode': 'SCALP',
                        'Bias (15m)': 'Bearish',
                        'SMA 20': 'Momentum DOWN',
                        'RSI': f"{latest['rsi']:.0f}",
                        'Momentum': 'Elephant Bar' if elephant_bar else 'Normal'
                    },
                    'warning': 'Thin Liquidity Above' if liquidity_data['hole_above'] else '',
                    'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
    
    # SWING LOGIC (15m, 1h, 4h)
    if is_swing:
        # LONG SWING
        if price_above_100 and sma20_above_100:
            if 40 <= latest['rsi'] <= 65:
                conf_score = 0
                entry_reasons = []
                
                if price_above_100:
                    conf_score += 2
                    entry_reasons.append("Price Above SMA 100")
                if sma20_above_100:
                    conf_score += 2
                    entry_reasons.append("SMA 20 Above 100 (Trend)")
                if 45 <= latest['rsi'] <= 60:
                    conf_score += 2
                    entry_reasons.append("RSI Perfect Zone")
                elif 40 <= latest['rsi'] <= 65:
                    conf_score += 1
                    entry_reasons.append("RSI Healthy")
                if elephant_bar and is_bullish_bar:
                    conf_score += 1
                    entry_reasons.append("Elephant Bar")
                if liquidity_data['hole_above']:
                    conf_score += 1
                    entry_reasons.append("Vacuum Above")
                
                if conf_score >= 6:
                    confidence = "🟢 HIGH"
                elif conf_score >= 4:
                    confidence = "🟡 MEDIUM"
                elif conf_score >= 2:
                    confidence = "🟠 CAUTION"
                else:
                    confidence = "⚪ WAIT"
                
                vacuum = " + VACUUM" if liquidity_data['hole_above'] else ""
                signals.append({
                    'symbol': symbol, 'timeframe': timeframe, 'exchange': '',
                    'strategy': f"🐢 LONG SWING{vacuum}",
                    'direction': 'UP', 'price': latest['price'],
                    'confidence': confidence,
                    'conf_score': conf_score,
                    'entry_reason': " | ".join(entry_reasons),
                    'conditions': {
                        'Mode': 'SWING',
                        'SMA 100': 'Price Above',
                        'SMA 20': 'Above 100',
                        'RSI': f"{latest['rsi']:.0f}",
                        'Momentum': 'Elephant Bar' if elephant_bar else 'Normal'
                    },
                    'warning': 'Thin Liquidity Below' if liquidity_data['hole_below'] else '',
                    'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # SHORT SWING
        if price_below_100 and sma20_below_100:
            if 35 <= latest['rsi'] <= 60:
                conf_score = 0
                entry_reasons = []
                
                if price_below_100:
                    conf_score += 2
                    entry_reasons.append("Price Below SMA 100")
                if sma20_below_100:
                    conf_score += 2
                    entry_reasons.append("SMA 20 Below 100 (Trend)")
                if 40 <= latest['rsi'] <= 55:
                    conf_score += 2
                    entry_reasons.append("RSI Perfect Zone")
                elif 35 <= latest['rsi'] <= 60:
                    conf_score += 1
                    entry_reasons.append("RSI Healthy")
                if elephant_bar and is_bearish_bar:
                    conf_score += 1
                    entry_reasons.append("Elephant Bar")
                if liquidity_data['hole_below']:
                    conf_score += 1
                    entry_reasons.append("Vacuum Below")
                
                if conf_score >= 6:
                    confidence = "🟢 HIGH"
                elif conf_score >= 4:
                    confidence = "🟡 MEDIUM"
                elif conf_score >= 2:
                    confidence = "🟠 CAUTION"
                else:
                    confidence = "⚪ WAIT"
                
                vacuum = " + VACUUM" if liquidity_data['hole_below'] else ""
                signals.append({
                    'symbol': symbol, 'timeframe': timeframe, 'exchange': '',
                    'strategy': f"🐢 SHORT SWING{vacuum}",
                    'direction': 'DOWN', 'price': latest['close'],
                    'confidence': confidence,
                    'conf_score': conf_score,
                    'entry_reason': " | ".join(entry_reasons),
                    'conditions': {
                        'Mode': 'SWING',
                        'SMA 100': 'Price Below',
                        'SMA 20': 'Below 100',
                        'RSI': f"{latest['rsi']:.0f}",
                        'Momentum': 'Elephant Bar' if elephant_bar else 'Normal'
                    },
                    'warning': 'Thin Liquidity Above' if liquidity_data['hole_above'] else '',
                    'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
    
    return signals

def scan_markets(exchanges_to_scan, symbols, timeframes):
    all_signals = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_scans = len(exchanges_to_scan) * len(symbols) * len(timeframes)
    current_scan = 0
    
    for exchange, exchange_name in exchanges_to_scan:
        for symbol in symbols:
            liquidity_data = check_liquidity(exchange, symbol)
            # Get higher timeframe for bias
            df_15m = fetch_ohlcv(exchange, symbol, '15m', 100)
            if df_15m is not None:
                df_15m = calculate_indicators(df_15m)
            
            for timeframe in timeframes:
                current_scan += 1
                progress_bar.progress(current_scan / total_scans)
                status_text.text(f"[{exchange_name}] {symbol} {timeframe}")
                try:
                    df = fetch_ohlcv(exchange, symbol, timeframe, 100)
                    if df is not None:
                        df = calculate_indicators(df)
                        if df is not None:
                            signals = detect_signals_trader_logic(df, df_15m, symbol, timeframe, liquidity_data)
                            for sig in signals:
                                sig['exchange'] = exchange_name
                                signal_key = f"{sig['exchange']}_{sig['symbol']}_{sig['timeframe']}_{sig['strategy']}"
                                existing_keys = [f"{h['exchange']}_{h['symbol']}_{h['timeframe']}_{h['strategy']}" for h in st.session_state.signal_history]
                                if signal_key not in existing_keys:
                                    st.session_state.signal_history.append(sig.copy())
                            all_signals.extend(signals)
                except:
                    continue
                time.sleep(0.05)
    
    progress_bar.empty()
    status_text.empty()
    if len(st.session_state.signal_history) > 100:
        st.session_state.signal_history = st.session_state.signal_history[-100:]
    return all_signals

def main():
    st.markdown("# ⚡ Scalper Pro Scanner")
    st.caption("📱 Scalper First, Swing Second | Pure Trader Logic")
    
    col1, col2 = st.columns(2)
    with col1:
        use_mexc = st.checkbox("📊 MEXC", value=True)
    with col2:
        use_gateio = st.checkbox("📊 Gate.io", value=False)
    
    if not use_mexc and not use_gateio:
        st.warning("⚠️ Select at least one exchange")
        st.stop()
    
    exchanges_to_scan = []
    if use_mexc:
        exchange_mexc, name_mexc = init_exchange('mexc')
        if exchange_mexc:
            exchanges_to_scan.append((exchange_mexc, 'MEXC'))
    if use_gateio:
        exchange_gate, name_gate = init_exchange('gateio')
        if exchange_gate:
            exchanges_to_scan.append((exchange_gate, 'Gate.io'))
    
    if not exchanges_to_scan:
        st.error("❌ Failed to connect")
        st.stop()
    
    exchange = exchanges_to_scan[0][0]
    
    with st.expander("⚙️ SETTINGS", expanded=False):
        try:
            markets = exchange.load_markets()
            usdt_pairs = [s for s in markets.keys() if s.endswith('/USDT') and markets[s]['active']]
            top_40 = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT', 'LTC/USDT', 'ATOM/USDT', 'XLM/USDT', 'ALGO/USDT', 'VET/USDT', 'ICP/USDT', 'FIL/USDT', 'HBAR/USDT', 'APT/USDT', 'ARB/USDT', 'OP/USDT', 'SUI/USDT', 'TIA/USDT', 'SEI/USDT', 'INJ/USDT', 'STX/USDT', 'RUNE/USDT', 'FTM/USDT', 'NEAR/USDT', 'AAVE/USDT', 'MKR/USDT', 'SNX/USDT', 'CRV/USDT', 'LDO/USDT', 'IMX/USDT', 'SAND/USDT', 'MANA/USDT', 'AXS/USDT']
            memecoins = ['PEPE/USDT', 'SHIB/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT', 'DOGE/USDT', 'MEME/USDT', 'WOJAK/USDT', 'TURBO/USDT', 'PEPE2/USDT', 'LADYS/USDT', 'BABYDOGE/USDT', 'ELON/USDT', 'KISHU/USDT', 'AKITA/USDT', 'SAMO/USDT', 'HOGE/USDT', 'SHIBAI/USDT', 'VOLT/USDT', 'CHEEMS/USDT', 'GROK/USDT', 'MYRO/USDT', 'MONG/USDT', 'NEIRO/USDT', 'MOG/USDT', 'TOSHI/USDT', 'BRETT/USDT', 'DEGEN/USDT', 'MEW/USDT', 'POPCAT/USDT', 'DOGS/USDT', 'NUBS/USDT', 'HAMMY/USDT', 'BONE/USDT', 'LEASH/USDT', 'SATS/USDT', 'RATS/USDT', 'ORDI/USDT', 'DORK/USDT', 'PONKE/USDT', 'BOZO/USDT', 'CATWIF/USDT', 'HOBBES/USDT', 'TRAC/USDT', 'BILLY/USDT', 'MAGA/USDT', 'BODEN/USDT', 'TREMP/USDT', 'COUPE/USDT', 'BITCOIN/USDT']
            trending = ['WLD/USDT', 'PYTH/USDT', 'JUP/USDT', 'STRK/USDT', 'DYM/USDT', 'PORTAL/USDT', 'ACE/USDT', 'NFP/USDT', 'AI/USDT', 'XAI/USDT', 'MANTA/USDT', 'ALT/USDT', 'JTO/USDT', 'PIXEL/USDT', 'SAGA/USDT', 'OMNI/USDT', 'MERL/USDT', 'REZ/USDT', 'BB/USDT', 'IO/USDT', 'ZK/USDT', 'ZRO/USDT', 'G/USDT', 'LISTA/USDT', 'BANANA/USDT', 'RENDER/USDT', 'FET/USDT', 'AGIX/USDT', 'OCEAN/USDT', 'GRT/USDT']
            top_40_available = [p for p in top_40 if p in usdt_pairs]
            memecoins_available = [p for p in memecoins if p in usdt_pairs]
            trending_available = [p for p in trending if p in usdt_pairs]
            default_120 = (top_40_available + memecoins_available + trending_available)[:120]
        except:
            usdt_pairs = []
            default_120 = []
        
        st.markdown("### 📊 Trading Pairs")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📈 Top 40", use_container_width=True):
                st.session_state.selected_pairs = top_40_available
        with col2:
            if st.button("🐸 Memes", use_container_width=True):
                st.session_state.selected_pairs = memecoins_available
        with col3:
            if st.button("🔥 Trending", use_container_width=True):
                st.session_state.selected_pairs = trending_available
        
        if st.button("⭐ ALL 120 COINS", use_container_width=True, type="primary"):
            st.session_state.selected_pairs = default_120
        
        if 'selected_pairs' not in st.session_state:
            st.session_state.selected_pairs = default_120
        
        selected_pairs = st.multiselect(f"Selected: {len(st.session_state.selected_pairs)} pairs", usdt_pairs, default=st.session_state.selected_pairs)
        
        st.markdown("### ⏱️ Timeframes")
        selected_timeframes = st.multiselect("Select", ['3m', '5m', '15m', '1h', '4h'], default=['3m', '5m', '15m'])
        
        st.markdown("### 🔄 Auto-Refresh")
        auto_refresh = st.toggle("Enable (60s)", value=st.session_state.auto_refresh_enabled)
        st.session_state.auto_refresh_enabled = auto_refresh
    
    scan_button = st.button("🔍 SCAN NOW", type="primary", use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Signals", len(st.session_state.signals))
    with col2:
        active_exchanges = []
        if use_mexc:
            active_exchanges.append("MEXC")
        if use_gateio:
            active_exchanges.append("Gate")
        st.metric("📡 Exchanges", " + ".join(active_exchanges))
    with col3:
        if st.session_state.last_update:
            st.metric("🕐 Updated", st.session_state.last_update.strftime("%H:%M"))
        else:
            st.metric("🕐 Updated", "Never")
    
    st.divider()
    
    if scan_button or (not st.session_state.signals and selected_pairs):
        if not selected_pairs:
            st.warning("⚠️ Select pairs")
        elif not selected_timeframes:
            st.warning("⚠️ Select timeframes")
        else:
            with st.spinner("🔍 Scanning..."):
                signals = scan_markets(exchanges_to_scan, selected_pairs, selected_timeframes)
                st.session_state.signals = signals
                st.session_state.last_update = datetime.now()
                st.rerun()
    
    if st.session_state.signals:
        st.markdown(f"### 📊 {len(st.session_state.signals)} Active Signals")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_conf = st.selectbox("Filter by Confidence", ["All", "🟢 HIGH Only", "🟡 MEDIUM+", "🟠 CAUTION+"])
        with col2:
            filter_type = st.selectbox("Filter by Type", ["All", "⚡ Scalps Only", "🐢 Swings Only"])
        
        # Apply filters
        filtered_signals = st.session_state.signals
        if filter_conf == "🟢 HIGH Only":
            filtered_signals = [s for s in filtered_signals if "🟢" in s.get('confidence', '')]
        elif filter_conf == "🟡 MEDIUM+":
            filtered_signals = [s for s in filtered_signals if "🟢" in s.get('confidence', '') or "🟡" in s.get('confidence', '')]
        elif filter_conf == "🟠 CAUTION+":
            filtered_signals = [s for s in filtered_signals if "⚪" not in s.get('confidence', '')]
        
        if filter_type == "⚡ Scalps Only":
            filtered_signals = [s for s in filtered_signals if "SCALP" in s['strategy']]
        elif filter_type == "🐢 Swings Only":
            filtered_signals = [s for s in filtered_signals if "SWING" in s['strategy']]
        
        st.caption(f"Showing {len(filtered_signals)} of {len(st.session_state.signals)} signals")
        
        # Display as table
        for signal in filtered_signals:
            if signal['direction'] == 'UP':
                direction_color = "#10b981"
            elif signal['direction'] == 'DOWN':
                direction_color = "#ef4444"
            else:
                direction_color = "#f59e0b"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {direction_color} 0%, {direction_color}dd 100%); 
                        padding: 1rem; border-radius: 12px; margin-bottom: 1rem; color: white;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="margin: 0;">{signal['exchange']} | {signal['symbol']}</h3>
                        <p style="margin: 0.3rem 0; font-size: 0.9rem; opacity: 0.9;">{signal['timeframe']} | ${signal['price']:.4f}</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="background: rgba(255,255,255,0.2); padding: 0.3rem 0.8rem; border-radius: 8px; font-size: 0.85rem; margin-bottom: 0.3rem;">
                            {signal['strategy']}
                        </div>
                        <div style="font-size: 1.3rem; font-weight: bold;">
                            {signal.get('confidence', 'N/A')}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Entry Reason Table
            st.markdown("**📋 ENTRY REASON:**")
            st.info(signal.get('entry_reason', 'No reason provided'))
            
            if signal.get('warning'):
                st.error(f"⚠️ {signal['warning']}")
            
            # Expandable details
            with st.expander("📊 See Full Details"):
                conditions = signal.get('conditions', {})
                col1, col2 = st.columns(2)
                with col1:
                    for key in list(conditions.keys())[:3]:
                        st.text(f"✓ {key}: {conditions[key]}")
                with col2:
                    for key in list(conditions.keys())[3:]:
                        st.text(f"✓ {key}: {conditions[key]}")
                
                st.text(f"Confidence Score: {signal.get('conf_score', 0)}/8")
            
            st.divider()
    else:
        st.info("👆 Tap **SCAN NOW**")
        st.markdown("""
        ### ⚡ Scalper Logic
        **LONG:** 15m bias up + Price > SMA 100 + SMA 20 momentum + RSI 45-60
        **SHORT:** 15m bias down + Price < SMA 100 + SMA 20 momentum + RSI 40-55
        
        **STAND DOWN:** Chop around SMA 100 or RSI extremes
        """)
    
    if st.session_state.signal_history:
        st.divider()
        with st.expander(f"📜 SIGNAL HISTORY ({len(st.session_state.signal_history)} Total)", expanded=False):
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.signal_history = []
                st.rerun()
            for idx, signal in enumerate(reversed(st.session_state.signal_history[-20:])):
                direction_emoji = "🟢" if signal['direction'] == 'UP' else "🔴" if signal['direction'] == 'DOWN' else "⚠️"
                st.markdown(f"**{direction_emoji} {signal['strategy']}** | {signal['exchange']} {signal['symbol']} - {signal['timeframe']} | ${signal['price']:.4f} | {signal['detected_at']}")
                if idx < 19:
                    st.markdown("---")
    
    st.divider()
    st.caption("⚡ Scalper First: 3m/5m with 15m bias | 🐢 Swing Second: 15m/1h/4h")
    st.caption("🧠 'Trade with SMA 100 bias, enter with SMA 20, filter with RSI'")

if __name__ == "__main__":
    main()
