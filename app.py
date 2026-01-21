# Version 23 - EXACT Trader Logic (NO DEVIATIONS)
import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import time

st.set_page_config(page_title="Expansion Scanner", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {padding: 0.5rem !important; max-width: 100% !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stButton>button {width: 100%; border-radius: 12px; height: 3.5em; font-weight: bold; font-size: 1rem;}
    @media (max-width: 768px) {
        .row-widget.stHorizontal {flex-direction: column !important;}
        div[data-testid="column"] {width: 100% !important; margin-bottom: 1rem;}
    }
</style>
""", unsafe_allow_html=True)

if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []
if 'daily_stats' not in st.session_state:
    st.session_state.daily_stats = {}
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
        return exchange, exchange_name
    except:
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
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        return df
    except:
        return None

def check_liquidity(exchange, symbol):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
        
        # Check for firewall (large orders blocking direction)
        threshold_above = mid_price * 1.02  # 2% above
        threshold_below = mid_price * 0.98  # 2% below
        
        ask_volume_above = sum([ask[1] for ask in orderbook['asks'] if ask[0] <= threshold_above])
        bid_volume_below = sum([bid[1] for bid in orderbook['bids'] if bid[0] >= threshold_below])
        
        avg_volume = (ask_volume_above + bid_volume_below) / 2
        
        firewall_above = ask_volume_above > avg_volume * 3  # 3x larger = firewall
        firewall_below = bid_volume_below > avg_volume * 3
        
        hole_above = ask_volume_above < avg_volume * 0.3  # Very thin = liquidity hole
        hole_below = bid_volume_below < avg_volume * 0.3
        
        return {
            'firewall_above': firewall_above,
            'firewall_below': firewall_below,
            'hole_above': hole_above,
            'hole_below': hole_below
        }
    except:
        return {'firewall_above': False, 'firewall_below': False, 'hole_above': False, 'hole_below': False}

def detect_expansion_signals(df, df_15m, symbol, timeframe, liquidity_data):
    if df is None or len(df) < 100:
        return []
    
    signals = []
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    if pd.isna(latest['sma_20']) or pd.isna(latest['sma_100']) or pd.isna(latest['rsi']):
        return []
    
    # STEP 1: CHECK 15M BIAS (MANDATORY FILTER)
    bias = "NEUTRAL"
    if df_15m is not None and len(df_15m) >= 100:
        latest_15m = df_15m.iloc[-1]
        prev_15m = df_15m.iloc[-2]
        
        if not pd.isna(latest_15m['sma_100']) and not pd.isna(latest_15m['sma_20']):
            price_above_100 = latest_15m['close'] > latest_15m['sma_100']
            price_below_100 = latest_15m['close'] < latest_15m['sma_100']
            sma20_rising = latest_15m['sma_20'] >= prev_15m['sma_20']
            sma20_falling = latest_15m['sma_20'] <= prev_15m['sma_20']
            rsi_bullish = latest_15m['rsi'] >= 50
            rsi_bearish = latest_15m['rsi'] <= 50
            
            if price_above_100 and (sma20_rising or abs(latest_15m['sma_20'] - prev_15m['sma_20']) < 0.001) and rsi_bullish:
                bias = "BULLISH"
            elif price_below_100 and (sma20_falling or abs(latest_15m['sma_20'] - prev_15m['sma_20']) < 0.001) and rsi_bearish:
                bias = "BEARISH"
            else:
                bias = "NEUTRAL"
    
    # If bias is NEUTRAL or unknown, output WAIT
    if bias == "NEUTRAL":
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': 'WAIT',
            'conviction': 'WAIT',
            'reason': '15m bias is NEUTRAL (chop around SMA100 or flat SMA20 or RSI 45-55). No trades allowed.',
            'price': latest['close'],
            'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'exchange': ''
        })
        return signals
    
    # STEP 2: DETECT STRUCTURES (Not entries, just context)
    # Squeeze detection
    sma_distance = abs(latest['sma_20'] - latest['sma_100']) / latest['sma_100'] * 100
    squeeze_present = sma_distance < 1.0
    
    # Crossover detection
    crossover_up = prev['sma_20'] <= prev['sma_100'] and latest['sma_20'] > latest['sma_100']
    crossover_down = prev['sma_20'] >= prev['sma_100'] and latest['sma_20'] < latest['sma_100']
    
    # STEP 3: DETECT EXPANSION (THE ONLY TRADEABLE EVENT)
    # Expansion = price moving AWAY from SMA structure, distance INCREASING
    prev_distance = abs(prev['close'] - prev['sma_100'])
    curr_distance = abs(latest['close'] - latest['sma_100'])
    prev2_distance = abs(prev2['close'] - prev2['sma_100'])
    
    expansion_happening = curr_distance > prev_distance and prev_distance > prev2_distance
    
    expansion_up = expansion_happening and latest['close'] > latest['sma_100']
    expansion_down = expansion_happening and latest['close'] < latest['sma_100']
    
    if not expansion_happening:
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': 'WAIT',
            'conviction': 'WAIT',
            'reason': 'No expansion detected. Price not moving decisively away from SMA structure.',
            'price': latest['close'],
            'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'exchange': ''
        })
        return signals
    
    # STEP 4: PRICE ACTION CONFIRMATION (Elephant bar OR Tail bar)
    # Elephant bar: Large range (>2x avg) with strong body (>65%)
    elephant_bar = False
    if not pd.isna(latest['avg_range']):
        elephant_bar = latest['range'] > latest['avg_range'] * 2.0 and latest['body'] > latest['range'] * 0.65
    
    # Tail bar: Clear rejection (wick >50% of range)
    tail_bar_bullish = latest['lower_wick'] > latest['range'] * 0.5 and latest['close'] > latest['open']
    tail_bar_bearish = latest['upper_wick'] > latest['range'] * 0.5 and latest['close'] < latest['open']
    
    candle_confirmation = elephant_bar or tail_bar_bullish or tail_bar_bearish
    
    if not candle_confirmation:
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': 'WAIT',
            'conviction': 'WAIT',
            'reason': 'Expansion present but no candle confirmation (need Elephant bar OR Tail bar).',
            'price': latest['close'],
            'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'exchange': ''
        })
        return signals
    
    # STEP 5: RSI POSITION (Context, not a signal)
    rsi_bullish_context = latest['rsi'] >= 50
    rsi_bearish_context = latest['rsi'] <= 50
    
    # STEP 6: LIQUIDITY CONTEXT
    firewall_blocking_long = liquidity_data['firewall_above']
    firewall_blocking_short = liquidity_data['firewall_below']
    liquidity_bonus_long = liquidity_data['hole_above']
    liquidity_bonus_short = liquidity_data['hole_below']
    
    # STEP 7: GENERATE SIGNALS
    # LONG SETUP
    if expansion_up and bias == "BULLISH":
        conviction = "WAIT"
        reason_parts = []
        
        # Check all conditions
        has_expansion = True
        has_candle = candle_confirmation
        has_rsi_aligned = rsi_bullish_context
        has_15m_bias = bias == "BULLISH"
        no_firewall = not firewall_blocking_long
        has_liquidity_bonus = liquidity_bonus_long
        
        reason_parts.append("Expansion UP confirmed (price moving away from SMA structure)")
        
        if elephant_bar:
            reason_parts.append("Elephant bar present (strong momentum)")
        elif tail_bar_bullish:
            reason_parts.append("Tail bar present (bullish rejection)")
        
        if has_rsi_aligned:
            reason_parts.append(f"RSI {latest['rsi']:.0f} (bullish context)")
        else:
            reason_parts.append(f"RSI {latest['rsi']:.0f} (bearish context - concern)")
        
        reason_parts.append("15m bias BULLISH (aligned)")
        
        if firewall_blocking_long:
            reason_parts.append("⚠️ FIREWALL ABOVE (large sell orders blocking path)")
            conviction = "CAUTION"
        elif has_liquidity_bonus:
            reason_parts.append("Liquidity hole above (clean path - BONUS)")
        else:
            reason_parts.append("Normal liquidity above")
        
        # Determine conviction
        if has_expansion and has_candle and has_rsi_aligned and has_15m_bias and no_firewall:
            if has_liquidity_bonus:
                conviction = "🟢 HIGH"
            else:
                conviction = "🟡 MEDIUM"
        elif has_expansion and has_candle and has_15m_bias:
            conviction = "🟠 CAUTION"
        else:
            conviction = "⚪ WAIT"
        
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': 'LONG',
            'conviction': conviction,
            'reason': " | ".join(reason_parts),
            'price': latest['close'],
            'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'exchange': ''
        })
    
    # SHORT SETUP
    if expansion_down and bias == "BEARISH":
        conviction = "WAIT"
        reason_parts = []
        
        has_expansion = True
        has_candle = candle_confirmation
        has_rsi_aligned = rsi_bearish_context
        has_15m_bias = bias == "BEARISH"
        no_firewall = not firewall_blocking_short
        has_liquidity_bonus = liquidity_bonus_short
        
        reason_parts.append("Expansion DOWN confirmed (price moving away from SMA structure)")
        
        if elephant_bar:
            reason_parts.append("Elephant bar present (strong momentum)")
        elif tail_bar_bearish:
            reason_parts.append("Tail bar present (bearish rejection)")
        
        if has_rsi_aligned:
            reason_parts.append(f"RSI {latest['rsi']:.0f} (bearish context)")
        else:
            reason_parts.append(f"RSI {latest['rsi']:.0f} (bullish context - concern)")
        
        reason_parts.append("15m bias BEARISH (aligned)")
        
        if firewall_blocking_short:
            reason_parts.append("⚠️ FIREWALL BELOW (large buy orders blocking path)")
            conviction = "CAUTION"
        elif has_liquidity_bonus:
            reason_parts.append("Liquidity hole below (clean path - BONUS)")
        else:
            reason_parts.append("Normal liquidity below")
        
        if has_expansion and has_candle and has_rsi_aligned and has_15m_bias and no_firewall:
            if has_liquidity_bonus:
                conviction = "🟢 HIGH"
            else:
                conviction = "🟡 MEDIUM"
        elif has_expansion and has_candle and has_15m_bias:
            conviction = "🟠 CAUTION"
        else:
            conviction = "⚪ WAIT"
        
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': 'SHORT',
            'conviction': conviction,
            'reason': " | ".join(reason_parts),
            'price': latest['close'],
            'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'exchange': ''
        })
    
    return signals

def update_daily_stats(signal):
    try:
        signal_date = signal.get('detected_at', '').split(' ')[0]
        if signal_date and signal['conviction'] != 'WAIT':
            if signal_date not in st.session_state.daily_stats:
                st.session_state.daily_stats[signal_date] = {
                    'total': 0, 'longs': 0, 'shorts': 0, 'high': 0, 'medium': 0, 'caution': 0
                }
            stats = st.session_state.daily_stats[signal_date]
            stats['total'] += 1
            if signal['direction'] == 'LONG':
                stats['longs'] += 1
            elif signal['direction'] == 'SHORT':
                stats['shorts'] += 1
            if '🟢' in signal['conviction']:
                stats['high'] += 1
            elif '🟡' in signal['conviction']:
                stats['medium'] += 1
            elif '🟠' in signal['conviction']:
                stats['caution'] += 1
    except:
        pass

def get_time_ago(timestamp_str):
    try:
        signal_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        now = datetime.now()
        diff = now - signal_time
        seconds = int(diff.total_seconds())
        if seconds < 10:
            return "🔴 NOW"
        elif seconds < 60:
            return f"🟠 {seconds}s ago"
        elif seconds < 3600:
            return f"🟡 {seconds // 60}m ago"
        elif seconds < 86400:
            return f"🟢 {seconds // 3600}h ago"
        else:
            return f"⚪ {seconds // 86400}d ago"
    except:
        return "⚪ Unknown"

def scan_markets(exchanges_to_scan, symbols, timeframes):
    all_signals = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_scans = len(exchanges_to_scan) * len(symbols) * len(timeframes)
    current_scan = 0
    
    for exchange, exchange_name in exchanges_to_scan:
        for symbol in symbols:
            liquidity_data = check_liquidity(exchange, symbol)
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
                            signals = detect_expansion_signals(df, df_15m, symbol, timeframe, liquidity_data)
                            for sig in signals:
                                sig['exchange'] = exchange_name
                                if sig['conviction'] != 'WAIT':
                                    signal_key = f"{sig['exchange']}_{sig['symbol']}_{sig['timeframe']}_{sig['direction']}"
                                    existing_keys = [f"{h['exchange']}_{h['symbol']}_{h['timeframe']}_{h['direction']}" for h in st.session_state.signal_history]
                                    if signal_key not in existing_keys:
                                        st.session_state.signal_history.append(sig.copy())
                                        update_daily_stats(sig)
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
    st.markdown("# ⚡ Expansion Scanner")
    st.caption("📱 Scalper First | Discretionary Edge | EXPANSION ONLY")
    
    col1, col2 = st.columns(2)
    with col1:
        use_mexc = st.checkbox("📊 MEXC", value=True)
    with col2:
        use_gateio = st.checkbox("📊 Gate.io", value=False)
    
    if not use_mexc and not use_gateio:
        st.warning("⚠️ Select exchange")
        st.stop()
    
    exchanges_to_scan = []
    if use_mexc:
        ex, nm = init_exchange('mexc')
        if ex:
            exchanges_to_scan.append((ex, 'MEXC'))
    if use_gateio:
        ex, nm = init_exchange('gateio')
        if ex:
            exchanges_to_scan.append((ex, 'Gate.io'))
    
    if not exchanges_to_scan:
        st.error("❌ Failed to connect")
        st.stop()
    
    exchange = exchanges_to_scan[0][0]
    
    with st.expander("⚙️ SETTINGS", expanded=False):
        try:
            markets = exchange.load_markets()
            usdt_pairs = [s for s in markets.keys() if s.endswith('/USDT') and markets[s]['active']]
            top_40 = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT']
            memecoins = ['PEPE/USDT', 'SHIB/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT', 'DOGE/USDT']
            
            top_40_available = [p for p in top_40 if p in usdt_pairs]
            memecoins_available = [p for p in memecoins if p in usdt_pairs]
            default_pairs = top_40_available[:10] + memecoins_available[:5]
        except:
            usdt_pairs = []
            default_pairs = []
        
        st.markdown("### 📊 Pairs")
        if 'selected_pairs' not in st.session_state:
            st.session_state.selected_pairs = default_pairs
        
        selected_pairs = st.multiselect(f"Selected: {len(st.session_state.selected_pairs)}", usdt_pairs, default=st.session_state.selected_pairs)
        
        st.markdown("### ⏱️ Timeframes (3m/5m only)")
        selected_timeframes = st.multiselect("Select", ['3m', '5m'], default=['3m', '5m'])
        
        st.markdown("### 🔄 Auto")
        auto_refresh = st.toggle("Enable (60s)", value=st.session_state.auto_refresh_enabled)
        st.session_state.auto_refresh_enabled = auto_refresh
    
    scan_button = st.button("🔍 SCAN", type="primary", use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        tradeable = len([s for s in st.session_state.signals if s['conviction'] != 'WAIT'])
        st.metric("🎯 Tradeable", tradeable)
    with col2:
        st.metric("📡 Exchange", " + ".join([e[1] for e in exchanges_to_scan]))
    with col3:
        if st.session_state.last_update:
            st.metric("🕐 Updated", st.session_state.last_update.strftime("%H:%M"))
        else:
            st.metric("🕐 Updated", "Never")
    
    st.divider()
    
    if scan_button or (not st.session_state.signals and selected_pairs):
        if not selected_pairs or not selected_timeframes:
            st.warning("⚠️ Select pairs and timeframes")
        else:
            with st.spinner("🔍 Scanning for EXPANSION..."):
                signals = scan_markets(exchanges_to_scan, selected_pairs, selected_timeframes)
                st.session_state.signals = signals
                st.session_state.last_update = datetime.now()
                st.rerun()
    
    if st.session_state.signals:
        tradeable_signals = [s for s in st.session_state.signals if s['conviction'] != 'WAIT']
        wait_signals = [s for s in st.session_state.signals if s['conviction'] == 'WAIT']
        
        if tradeable_signals:
            st.markdown(f"### 📊 {len(tradeable_signals)} TRADEABLE Setups")
            
            for signal in tradeable_signals:
                if signal['direction'] == 'LONG':
                    color = "#10b981"
                else:
                    color = "#ef4444"
                
                time_ago = get_time_ago(signal['detected_at'])
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                            padding: 1rem; border-radius: 12px; margin-bottom: 1rem; color: white;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <h3 style="margin: 0;">{signal['exchange']} | {signal['symbol']}</h3>
                            <p style="margin: 0.3rem 0; font-size: 0.9rem;">{signal['timeframe']} | ${signal['price']:.4f}</p>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.3rem; font-weight: bold; margin-bottom: 0.3rem;">
                                {signal['conviction']}
                            </div>
                            <div style="background: rgba(0,0,0,0.3); padding: 0.3rem 0.6rem; border-radius: 6px; font-size: 0.85rem;">
                                ⏱️ {time_ago}
                            </div>
                        </div>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); padding: 0.5rem 0.8rem; border-radius: 8px; font-size: 0.9rem; margin-top: 0.5rem;">
                        {signal['direction']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"**REASON:** {signal['reason']}")
                st.divider()
        
        if wait_signals:
            with st.expander(f"⚪ {len(wait_signals)} WAIT Signals (Not Tradeable)", expanded=False):
                for signal in wait_signals[:10]:
                    st.markdown(f"**{signal['symbol']} - {signal['timeframe']}**")
                    st.caption(signal['reason'])
                    st.markdown("---")
    else:
        st.info("👆 Tap **SCAN**")
        st.markdown("""
        ### 🧠 EXPANSION LOGIC
        **ONLY tradeable event:** Price moving decisively AWAY from SMA structure
        
        **Required:**
        1. 15m bias aligned (BULLISH or BEARISH)
        2. Expansion confirmed (distance increasing)
        3. Elephant bar OR Tail bar
        4. RSI context aligned
        5. No firewall blocking direction
        
        **If ANY condition missing:** Output = WAIT
        """)
    
    st.divider()
    st.caption("⚡ EXPANSION ONLY | No predictions | No forced trades | Pure discretionary edge")

if __name__ == "__main__":
    main()
