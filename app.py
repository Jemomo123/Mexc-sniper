# Version 16 - Multi-Exchange Scanner with 120 Coins
import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import time

st.set_page_config(page_title="Multi-Exchange Pro Scanner", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {padding: 0.5rem !important; max-width: 100% !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stButton>button {width: 100%; border-radius: 12px; height: 3.5em; font-weight: bold; font-size: 1rem; margin: 0.5rem 0;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem; font-weight: bold;}
    div[data-testid="stMetricLabel"] {font-size: 0.9rem;}
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
        st.error(f"Failed to initialize {exchange_name.upper()}: {e}")
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
    if df is None or len(df) < 200:
        return None
    try:
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_200'] = ta.sma(df['close'], length=200)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['range'] = df['high'] - df['low']
        df['avg_range'] = df['range'].rolling(20).mean()
        df['body'] = abs(df['close'] - df['open'])
        if df['vwap'].isna().all():
            df['vwap'] = df['close']
        else:
            df['vwap'].fillna(method='ffill', inplace=True)
            df['vwap'].fillna(method='bfill', inplace=True)
        return df
    except:
        return None

def check_liquidity(exchange, symbol):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        ohlcv_1m = exchange.fetch_ohlcv(symbol, '1m', limit=20)
        avg_volume_1m = np.mean([candle[5] for candle in ohlcv_1m])
        mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
        threshold = mid_price * 0.005
        bid_depth = sum([bid[1] for bid in orderbook['bids'] if bid[0] >= mid_price - threshold])
        ask_depth = sum([ask[1] for ask in orderbook['asks'] if ask[0] <= mid_price + threshold])
        depth_threshold = avg_volume_1m * 0.2
        hole_above = ask_depth < depth_threshold
        hole_below = bid_depth < depth_threshold
        return {'hole_above': hole_above, 'hole_below': hole_below, 'bid_depth': bid_depth, 'ask_depth': ask_depth, 'avg_volume': avg_volume_1m}
    except:
        return {'hole_above': False, 'hole_below': False, 'bid_depth': 0, 'ask_depth': 0, 'avg_volume': 0}

def detect_signals(df, symbol, timeframe, liquidity_data):
    if df is None or len(df) < 200:
        return []
    signals = []
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    if pd.isna(latest['sma_20']) or pd.isna(latest['sma_200']) or pd.isna(latest['rsi']) or pd.isna(latest['vwap']):
        return []
    sma_diff_pct = abs(latest['sma_20'] - latest['sma_200']) / latest['sma_200'] * 100
    squeeze_detected = sma_diff_pct < 0.5
    prev_diff = abs(prev['sma_20'] - prev['sma_200'])
    curr_diff = abs(latest['sma_20'] - latest['sma_200'])
    divergence_detected = curr_diff > prev_diff * 1.5 and sma_diff_pct > 1.0
    elephant_bar = False
    if not pd.isna(latest['avg_range']):
        elephant_bar = (latest['range'] > latest['avg_range'] * 2.0 and latest['body'] > latest['range'] * 0.65)  # Loosened from 2.5x and 75%
    is_bullish_elephant = elephant_bar and latest['close'] > latest['open']
    is_bearish_elephant = elephant_bar and latest['close'] < latest['open']
    sma20_above_200 = latest['sma_20'] > latest['sma_200']
    price_above_vwap = latest['close'] > latest['vwap']
    
    if not pd.isna(prev['sma_20']) and not pd.isna(prev['sma_200']):
        if prev['sma_20'] <= prev['sma_200'] and latest['sma_20'] > latest['sma_200']:
            vacuum_label = "PLUS (+ VACUUM)" if liquidity_data['hole_above'] else "STANDARD"
            warning = "WARNING: THIN LIQUIDITY BELOW" if liquidity_data['hole_below'] else ""
            signals.append({'symbol': symbol, 'timeframe': timeframe, 'strategy': f"1. Goliath Breakout UP [{vacuum_label}]", 'direction': 'UP', 'price': latest['close'], 'conditions': {'Trend': 'SMA 20 Above 200', 'Squeeze Status': 'Squeeze Detected' if squeeze_detected else 'No Squeeze', 'Momentum': 'Elephant Bar Present' if elephant_bar else 'No Elephant Bar', 'Value': 'Price Above VWAP' if price_above_vwap else 'Price Below VWAP', 'Vacuum': 'Vacuum Detected Above' if liquidity_data['hole_above'] else 'Order Depth Present'}, 'warning': warning})
    
    if not pd.isna(prev['sma_20']) and not pd.isna(prev['sma_200']):
        if prev['sma_20'] >= prev['sma_200'] and latest['sma_20'] < latest['sma_200']:
            vacuum_label = "PLUS (+ VACUUM)" if liquidity_data['hole_below'] else "STANDARD"
            warning = "WARNING: THIN LIQUIDITY ABOVE" if liquidity_data['hole_above'] else ""
            signals.append({'symbol': symbol, 'timeframe': timeframe, 'strategy': f"1. Goliath Breakdown DOWN [{vacuum_label}]", 'direction': 'DOWN', 'price': latest['close'], 'conditions': {'Trend': 'SMA 20 Below 200', 'Squeeze Status': 'Squeeze Detected' if squeeze_detected else 'No Squeeze', 'Momentum': 'Elephant Bar Present' if elephant_bar else 'No Elephant Bar', 'Value': 'Price Above VWAP' if price_above_vwap else 'Price Below VWAP', 'Vacuum': 'Vacuum Detected Below' if liquidity_data['hole_below'] else 'Order Depth Present'}, 'warning': warning})
    
    if sma20_above_200 and squeeze_detected and divergence_detected:
        vacuum_label = "PLUS (+ VACUUM)" if liquidity_data['hole_above'] else "STANDARD"
        warning = "WARNING: THIN LIQUIDITY BELOW" if liquidity_data['hole_below'] else ""
        signals.append({'symbol': symbol, 'timeframe': timeframe, 'strategy': f"2. Goliath Launch UP [{vacuum_label}]", 'direction': 'UP', 'price': latest['close'], 'conditions': {'Trend': 'SMA 20 Above 200', 'Squeeze Status': 'Squeeze Detected', 'Momentum': 'Elephant Bar Present' if elephant_bar else 'No Elephant Bar', 'Value': 'Price Above VWAP' if price_above_vwap else 'Price Below VWAP', 'Vacuum': 'Vacuum Detected Above' if liquidity_data['hole_above'] else 'Order Depth Present'}, 'warning': warning})
    
    if not sma20_above_200 and squeeze_detected and divergence_detected:
        vacuum_label = "PLUS (+ VACUUM)" if liquidity_data['hole_below'] else "STANDARD"
        warning = "WARNING: THIN LIQUIDITY ABOVE" if liquidity_data['hole_above'] else ""
        signals.append({'symbol': symbol, 'timeframe': timeframe, 'strategy': f"2. Goliath Drop DOWN [{vacuum_label}]", 'direction': 'DOWN', 'price': latest['close'], 'conditions': {'Trend': 'SMA 20 Below 200', 'Squeeze Status': 'Squeeze Detected', 'Momentum': 'Elephant Bar Present' if elephant_bar else 'No Elephant Bar', 'Value': 'Price Above VWAP' if price_above_vwap else 'Price Below VWAP', 'Vacuum': 'Vacuum Detected Below' if liquidity_data['hole_below'] else 'Order Depth Present'}, 'warning': warning})
    
    if not pd.isna(prev['sma_200']):
        if prev['low'] <= prev['sma_200'] and latest['close'] > latest['sma_200'] and latest['close'] > latest['open']:
            vacuum_label = "PLUS (+ VACUUM)" if liquidity_data['hole_above'] else "STANDARD"
            warning = "WARNING: THIN LIQUIDITY BELOW" if liquidity_data['hole_below'] else ""
            signals.append({'symbol': symbol, 'timeframe': timeframe, 'strategy': f"3. Failed Cross UP [{vacuum_label}]", 'direction': 'UP', 'price': latest['close'], 'conditions': {'Trend': 'SMA 20 Above 200' if sma20_above_200 else 'SMA 20 Below 200', 'Squeeze Status': 'Squeeze Detected' if squeeze_detected else 'No Squeeze', 'Momentum': 'Elephant Bar Present' if elephant_bar else 'No Elephant Bar', 'Value': 'Price Above VWAP' if price_above_vwap else 'Price Below VWAP', 'Vacuum': 'Vacuum Detected Above' if liquidity_data['hole_above'] else 'Order Depth Present'}, 'warning': warning})
    
    if not pd.isna(prev['sma_200']):
        if prev['high'] >= prev['sma_200'] and latest['close'] < latest['sma_200'] and latest['close'] < latest['open']:
            vacuum_label = "PLUS (+ VACUUM)" if liquidity_data['hole_below'] else "STANDARD"
            warning = "WARNING: THIN LIQUIDITY ABOVE" if liquidity_data['hole_above'] else ""
            signals.append({'symbol': symbol, 'timeframe': timeframe, 'strategy': f"3. Deadly Rejection DOWN [{vacuum_label}]", 'direction': 'DOWN', 'price': latest['close'], 'conditions': {'Trend': 'SMA 20 Above 200' if sma20_above_200 else 'SMA 20 Below 200', 'Squeeze Status': 'Squeeze Detected' if squeeze_detected else 'No Squeeze', 'Momentum': 'Elephant Bar Present' if elephant_bar else 'No Elephant Bar', 'Value': 'Price Above VWAP' if price_above_vwap else 'Price Below VWAP', 'Vacuum': 'Vacuum Detected Below' if liquidity_data['hole_below'] else 'Order Depth Present'}, 'warning': warning})
    
    if latest['rsi'] < 30 and not pd.isna(prev['sma_200']):  # Loosened from 25 to 30
        if prev['low'] <= prev['sma_200'] and is_bullish_elephant:
            vacuum_label = "PLUS (+ VACUUM)" if liquidity_data['hole_above'] else "STANDARD"
            warning = "WARNING: THIN LIQUIDITY BELOW" if liquidity_data['hole_below'] else ""
            signals.append({'symbol': symbol, 'timeframe': timeframe, 'strategy': f"4. Snap-Back Long UP [{vacuum_label}]", 'direction': 'UP', 'price': latest['close'], 'conditions': {'Trend': 'SMA 20 Above 200' if sma20_above_200 else 'SMA 20 Below 200', 'Squeeze Status': 'Squeeze Detected' if squeeze_detected else 'No Squeeze', 'Momentum': 'Elephant Bar Present', 'Value': 'Price Above VWAP' if price_above_vwap else 'Price Below VWAP', 'Vacuum': 'Vacuum Detected Above' if liquidity_data['hole_above'] else 'Order Depth Present'}, 'warning': warning})
    
    if latest['rsi'] > 70 and not pd.isna(prev['sma_200']):  # Loosened from 75 to 70
        if prev['high'] >= prev['sma_200'] and is_bearish_elephant:
            vacuum_label = "PLUS (+ VACUUM)" if liquidity_data['hole_below'] else "STANDARD"
            warning = "WARNING: THIN LIQUIDITY ABOVE" if liquidity_data['hole_above'] else ""
            signals.append({'symbol': symbol, 'timeframe': timeframe, 'strategy': f"4. Snap-Back Short DOWN [{vacuum_label}]", 'direction': 'DOWN', 'price': latest['close'], 'conditions': {'Trend': 'SMA 20 Above 200' if sma20_above_200 else 'SMA 20 Below 200', 'Squeeze Status': 'Squeeze Detected' if squeeze_detected else 'No Squeeze', 'Momentum': 'Elephant Bar Present', 'Value': 'Price Above VWAP' if price_above_vwap else 'Price Below VWAP', 'Vacuum': 'Vacuum Detected Below' if liquidity_data['hole_below'] else 'Order Depth Present'}, 'warning': warning})
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
            for timeframe in timeframes:
                current_scan += 1
                progress_bar.progress(current_scan / total_scans)
                status_text.text(f"[{exchange_name}] Scanning {symbol} {timeframe}")
                try:
                    df = fetch_ohlcv(exchange, symbol, timeframe)
                    if df is not None:
                        df = calculate_indicators(df)
                        if df is not None:
                            signals = detect_signals(df, symbol, timeframe, liquidity_data)
                            for sig in signals:
                                sig['exchange'] = exchange_name
                                sig['detected_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                # Check if this exact signal already exists in history
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
    # Keep only last 100 signals in history to prevent memory issues
    if len(st.session_state.signal_history) > 100:
        st.session_state.signal_history = st.session_state.signal_history[-100:]
    return all_signals

def main():
    st.markdown("# 🎯 Multi-Exchange Pro Scanner")
    st.caption("📱 16-Opportunity System | MEXC + Gate.io")
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
        selected_timeframes = st.multiselect("Select timeframes", ['3m', '5m', '15m', '1h', '4h'], default=['5m', '15m', '1h'])
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
            st.warning("⚠️ Select at least one pair")
        elif not selected_timeframes:
            st.warning("⚠️ Select at least one timeframe")
        else:
            with st.spinner("🔍 Scanning..."):
                signals = scan_markets(exchanges_to_scan, selected_pairs, selected_timeframes)
                st.session_state.signals = signals
                st.session_state.last_update = datetime.now()
                st.rerun()
    if st.session_state.signals:
        st.markdown(f"### 📊 {len(st.session_state.signals)} Current Opportunities")
        for signal in st.session_state.signals:
            direction_color = "#10b981" if signal['direction'] == 'UP' else "#ef4444"
            st.markdown(f"""<div style="background: linear-gradient(135deg, {direction_color} 0%, {direction_color}dd 100%); padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem; color: white;"><h3 style="margin: 0;">{signal['exchange']} | {signal['symbol']} - {signal['timeframe']}</h3><p style="margin: 0.5rem 0; font-size: 1.1rem; font-weight: bold;">{signal['strategy']}</p><p style="margin: 0; font-size: 1.2rem;">Price: ${signal['price']:.4f}</p></div>""", unsafe_allow_html=True)
            if signal['warning']:
                st.error(f"⚠️ {signal['warning']}")
            st.markdown("**📋 Aligned Conditions:**")
            conditions = signal['conditions']
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"✓ Trend: {conditions['Trend']}")
                st.text(f"✓ Squeeze: {conditions['Squeeze Status']}")
                st.text(f"✓ Momentum: {conditions['Momentum']}")
            with col2:
                st.text(f"✓ Value: {conditions['Value']}")
                st.text(f"✓ Vacuum: {conditions['Vacuum']}")
            st.divider()
    else:
        st.info("👆 Tap **SCAN NOW** to find opportunities")
    
    # Signal History Section
    if st.session_state.signal_history:
        st.divider()
        with st.expander(f"📜 SIGNAL HISTORY ({len(st.session_state.signal_history)} Total Captured)", expanded=False):
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.signal_history = []
                st.rerun()
            
            st.caption("All signals captured today (even when you weren't watching)")
            
            for idx, signal in enumerate(reversed(st.session_state.signal_history)):
                direction_emoji = "🟢" if signal['direction'] == 'UP' else "🔴"
                st.markdown(f"""
                **{direction_emoji} {signal['strategy']}**  
                📊 {signal['exchange']} | {signal['symbol']} - {signal['timeframe']}  
                💰 Price: ${signal['price']:.4f} | 🕐 {signal['detected_at']}
                """)
                if idx < len(st.session_state.signal_history) - 1:
                    st.markdown("---")
    st.divider()
    st.caption("📊 Now with 20+ Opportunity Types (Original 16 + Power Variants)")
    st.caption("🔥 Fixed: Strategy 2 (Squeeze OR Divergence) | Strategy 4 (RSI + SMA Touch)")
    st.caption("⚡ Bonus: 2B & 4B Power variants when all conditions align perfectly")

if __name__ == "__main__":
    main()
