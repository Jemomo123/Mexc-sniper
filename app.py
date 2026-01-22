# Version 24 - Expansion Edge (SQZ + Cross Independent Logic)
import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import time

st.set_page_config(page_title="Expansion Edge Scanner", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {padding: 0.5rem !important; max-width: 100% !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stButton>button {width: 100%; border-radius: 12px; height: 3.5em; font-weight: bold;}
    @media (max-width: 768px) {
        .row-widget.stHorizontal {flex-direction: column !important;}
        div[data-testid="column"] {width: 100% !important;}
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
        threshold_above = mid_price * 1.02
        threshold_below = mid_price * 0.98
        ask_volume = sum([ask[1] for ask in orderbook['asks'] if ask[0] <= threshold_above])
        bid_volume = sum([bid[1] for bid in orderbook['bids'] if bid[0] >= threshold_below])
        avg_volume = (ask_volume + bid_volume) / 2
        firewall_above = ask_volume > avg_volume * 3
        firewall_below = bid_volume > avg_volume * 3
        void_above = ask_volume < avg_volume * 0.3
        void_below = bid_volume < avg_volume * 0.3
        return {
            'firewall_above': firewall_above,
            'firewall_below': firewall_below,
            'void_above': void_above,
            'void_below': void_below
        }
    except:
        return {'firewall_above': False, 'firewall_below': False, 'void_above': False, 'void_below': False}

def detect_expansion_edge(df, df_15m, symbol, timeframe, liquidity_data):
    if df is None or len(df) < 100:
        return []
    
    signals = []
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    if pd.isna(latest['sma_20']) or pd.isna(latest['sma_100']) or pd.isna(latest['rsi']):
        return []
    
    # STEP 1: 15M BIAS CHECK
    bias_15m = "NEUTRAL"
    if df_15m is not None and len(df_15m) >= 100:
        latest_15m = df_15m.iloc[-1]
        if not pd.isna(latest_15m['sma_100']):
            if latest_15m['close'] > latest_15m['sma_100']:
                bias_15m = "BULLISH"
            elif latest_15m['close'] < latest_15m['sma_100']:
                bias_15m = "BEARISH"
    
    # STEP 2: DETECT STRUCTURES (Not entries)
    sma_distance_pct = abs(latest['sma_20'] - latest['sma_100']) / latest['sma_100'] * 100
    squeeze_present = sma_distance_pct < 1.5
    
    crossover_up = prev['sma_20'] <= prev['sma_100'] and latest['sma_20'] > latest['sma_100']
    crossover_down = prev['sma_20'] >= prev['sma_100'] and latest['sma_20'] < latest['sma_100']
    
    # STEP 3: DETECT EXPANSION (Price moving away with intent)
    prev_distance = abs(prev['close'] - prev['sma_100'])
    curr_distance = abs(latest['close'] - latest['sma_100'])
    prev2_distance = abs(prev2['close'] - prev2['sma_100'])
    
    expansion_confirmed = curr_distance > prev_distance and prev_distance > prev2_distance
    
    expansion_direction = None
    if expansion_confirmed:
        if latest['close'] > latest['sma_100']:
            expansion_direction = "UP"
        else:
            expansion_direction = "DOWN"
    
    if not expansion_confirmed:
        signals.append({
            'symbol': symbol, 'timeframe': timeframe, 'exchange': '',
            'direction': 'WAIT', 'conviction': 'WAIT', 'position_size': '0R',
            'reason': 'No expansion detected. Price not moving away from SMA zone with intent.',
            'price': latest['close'], 'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        return signals
    
    # STEP 4: EXPANSION CONFIRMATION (Elephant OR Tail bar)
    elephant_bar = False
    if not pd.isna(latest['avg_range']):
        elephant_bar = latest['range'] > latest['avg_range'] * 2.0 and latest['body'] > latest['range'] * 0.65
    
    tail_bar_bullish = latest['lower_wick'] > latest['range'] * 0.5 and latest['close'] > latest['open']
    tail_bar_bearish = latest['upper_wick'] > latest['range'] * 0.5 and latest['close'] < latest['open']
    
    bar_confirmation = elephant_bar or tail_bar_bullish or tail_bar_bearish
    
    if not bar_confirmation:
        signals.append({
            'symbol': symbol, 'timeframe': timeframe, 'exchange': '',
            'direction': 'WAIT', 'conviction': 'WAIT', 'position_size': '0R',
            'reason': 'Expansion present but NO elephant or tail bar confirmation. NO TRADE.',
            'price': latest['close'], 'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        return signals
    
    # STEP 5: BUILD SIGNALS
    if expansion_direction == "UP":
        reason_parts = []
        conviction = "WAIT"
        position_size = "0R"
        
        # Context
        if squeeze_present and not crossover_up:
            reason_parts.append("Expansion from SQUEEZE (no cross needed)")
        elif crossover_up:
            reason_parts.append("Expansion after CROSSOVER UP")
        else:
            reason_parts.append("Expansion UP confirmed")
        
        # Bar type
        if elephant_bar:
            reason_parts.append("Elephant bar present")
        elif tail_bar_bullish:
            reason_parts.append("Tail bar (bullish rejection)")
        
        # RSI
        if latest['rsi'] >= 50:
            reason_parts.append(f"RSI {latest['rsi']:.0f} (supports UP)")
        else:
            reason_parts.append(f"RSI {latest['rsi']:.0f} (conflicts - bearish)")
        
        # 15m bias
        if bias_15m == "BULLISH":
            reason_parts.append("15m bias BULLISH (aligned)")
        elif bias_15m == "BEARISH":
            reason_parts.append("15m bias BEARISH (conflict)")
        else:
            reason_parts.append("15m bias NEUTRAL")
        
        # Liquidity
        if liquidity_data['firewall_above']:
            reason_parts.append("⚠️ FIREWALL above (heavy resistance)")
        elif liquidity_data['void_above']:
            reason_parts.append("✅ VOID above (clean path)")
        else:
            reason_parts.append("Normal liquidity above")
        
        # CONVICTION LOGIC
        rsi_aligned = latest['rsi'] >= 50
        bias_aligned = bias_15m == "BULLISH"
        no_firewall = not liquidity_data['firewall_above']
        has_void = liquidity_data['void_above']
        
        if expansion_confirmed and bar_confirmation and rsi_aligned and bias_aligned and no_firewall:
            if has_void:
                conviction = "🟢 HIGH A"
                position_size = "1.0R"
            else:
                conviction = "🟢 HIGH B"
                position_size = "0.6-0.8R"
        elif expansion_confirmed and bar_confirmation and bias_aligned:
            conviction = "🟠 CAUTION"
            position_size = "0.3-0.5R"
        else:
            conviction = "⚪ WAIT"
            position_size = "0R"
        
        signals.append({
            'symbol': symbol, 'timeframe': timeframe, 'exchange': '',
            'direction': 'LONG', 'conviction': conviction, 'position_size': position_size,
            'reason': " | ".join(reason_parts),
            'price': latest['close'], 'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    elif expansion_direction == "DOWN":
        reason_parts = []
        conviction = "WAIT"
        position_size = "0R"
        
        if squeeze_present and not crossover_down:
            reason_parts.append("Expansion from SQUEEZE (no cross needed)")
        elif crossover_down:
            reason_parts.append("Expansion after CROSSOVER DOWN")
        else:
            reason_parts.append("Expansion DOWN confirmed")
        
        if elephant_bar:
            reason_parts.append("Elephant bar present")
        elif tail_bar_bearish:
            reason_parts.append("Tail bar (bearish rejection)")
        
        if latest['rsi'] <= 50:
            reason_parts.append(f"RSI {latest['rsi']:.0f} (supports DOWN)")
        else:
            reason_parts.append(f"RSI {latest['rsi']:.0f} (conflicts - bullish)")
        
        if bias_15m == "BEARISH":
            reason_parts.append("15m bias BEARISH (aligned)")
        elif bias_15m == "BULLISH":
            reason_parts.append("15m bias BULLISH (conflict)")
        else:
            reason_parts.append("15m bias NEUTRAL")
        
        if liquidity_data['firewall_below']:
            reason_parts.append("⚠️ FIREWALL below (heavy support)")
        elif liquidity_data['void_below']:
            reason_parts.append("✅ VOID below (clean path)")
        else:
            reason_parts.append("Normal liquidity below")
        
        rsi_aligned = latest['rsi'] <= 50
        bias_aligned = bias_15m == "BEARISH"
        no_firewall = not liquidity_data['firewall_below']
        has_void = liquidity_data['void_below']
        
        if expansion_confirmed and bar_confirmation and rsi_aligned and bias_aligned and no_firewall:
            if has_void:
                conviction = "🟢 HIGH A"
                position_size = "1.0R"
            else:
                conviction = "🟢 HIGH B"
                position_size = "0.6-0.8R"
        elif expansion_confirmed and bar_confirmation and bias_aligned:
            conviction = "🟠 CAUTION"
            position_size = "0.3-0.5R"
        else:
            conviction = "⚪ WAIT"
            position_size = "0R"
        
        signals.append({
            'symbol': symbol, 'timeframe': timeframe, 'exchange': '',
            'direction': 'SHORT', 'conviction': conviction, 'position_size': position_size,
            'reason': " | ".join(reason_parts),
            'price': latest['close'], 'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return signals

def update_daily_stats(signal):
    try:
        if signal['conviction'] == 'WAIT':
            return
        signal_date = signal.get('detected_at', '').split(' ')[0]
        if signal_date:
            if signal_date not in st.session_state.daily_stats:
                st.session_state.daily_stats[signal_date] = {
                    'total': 0, 'longs': 0, 'shorts': 0, 'high_a': 0, 'high_b': 0, 'caution': 0
                }
            stats = st.session_state.daily_stats[signal_date]
            stats['total'] += 1
            if signal['direction'] == 'LONG':
                stats['longs'] += 1
            elif signal['direction'] == 'SHORT':
                stats['shorts'] += 1
            if 'HIGH A' in signal['conviction']:
                stats['high_a'] += 1
            elif 'HIGH B' in signal['conviction']:
                stats['high_b'] += 1
            elif 'CAUTION' in signal['conviction']:
                stats['caution'] += 1
    except:
        pass

def get_time_ago(timestamp_str):
    try:
        signal_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        diff = (datetime.now() - signal_time).total_seconds()
        if diff < 10:
            return "🔴 NOW"
        elif diff < 60:
            return f"🟠 {int(diff)}s"
        elif diff < 3600:
            return f"🟡 {int(diff // 60)}m"
        elif diff < 86400:
            return f"🟢 {int(diff // 3600)}h"
        else:
            return f"⚪ {int(diff // 86400)}d"
    except:
        return "⚪ ?"

def scan_markets(exchanges_to_scan, symbols, timeframes):
    all_signals = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(exchanges_to_scan) * len(symbols) * len(timeframes)
    current = 0
    
    for exchange, exchange_name in exchanges_to_scan:
        for symbol in symbols:
            liquidity_data = check_liquidity(exchange, symbol)
            df_15m = fetch_ohlcv(exchange, symbol, '15m', 100)
            if df_15m is not None:
                df_15m = calculate_indicators(df_15m)
            
            for timeframe in timeframes:
                current += 1
                progress_bar.progress(current / total)
                status_text.text(f"[{exchange_name}] {symbol} {timeframe}")
                try:
                    df = fetch_ohlcv(exchange, symbol, timeframe, 100)
                    if df is not None:
                        df = calculate_indicators(df)
                        if df is not None:
                            signals = detect_expansion_edge(df, df_15m, symbol, timeframe, liquidity_data)
                            for sig in signals:
                                sig['exchange'] = exchange_name
                                if sig['conviction'] != 'WAIT':
                                    key = f"{sig['exchange']}_{sig['symbol']}_{sig['timeframe']}_{sig['direction']}"
                                    existing = [f"{h['exchange']}_{h['symbol']}_{h['timeframe']}_{h['direction']}" for h in st.session_state.signal_history]
                                    if key not in existing:
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
    st.markdown("# ⚡ Expansion Edge Scanner")
    st.caption("📱 SQZ + Cross Independent | Expansion Only | Scalper First")
    
    col1, col2 = st.columns(2)
    with col1:
        use_mexc = st.checkbox("📊 MEXC", value=True)
    with col2:
        use_gateio = st.checkbox("📊 Gate.io", value=False)
    
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
        st.error("❌ No exchange")
        st.stop()
    
    exchange = exchanges_to_scan[0][0]
    
    with st.expander("⚙️ SETTINGS", expanded=False):
        try:
            markets = exchange.load_markets()
            usdt_pairs = [s for s in markets.keys() if s.endswith('/USDT') and markets[s]['active']]
            top = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'AVAX/USDT', 'MATIC/USDT', 'ARB/USDT', 'OP/USDT']
            memes = ['PEPE/USDT', 'SHIB/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT', 'DOGE/USDT', 'MEME/USDT']
            default = [p for p in top + memes if p in usdt_pairs]
        except:
            usdt_pairs = []
            default = []
        
        if 'selected_pairs' not in st.session_state:
            st.session_state.selected_pairs = default
        
        selected_pairs = st.multiselect(f"Pairs ({len(st.session_state.selected_pairs)})", usdt_pairs, default=st.session_state.selected_pairs)
        selected_timeframes = st.multiselect("Timeframes (3m/5m main)", ['3m', '5m'], default=['3m', '5m'])
        auto_refresh = st.toggle("Auto (60s)", value=st.session_state.auto_refresh_enabled)
        st.session_state.auto_refresh_enabled = auto_refresh
    
    scan_button = st.button("🔍 SCAN EXPANSION", type="primary", use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        tradeable = len([s for s in st.session_state.signals if s.get('conviction', 'WAIT') != 'WAIT'])
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
        tradeable = [s for s in st.session_state.signals if s.get('conviction', 'WAIT') != 'WAIT']
        wait = [s for s in st.session_state.signals if s.get('conviction', 'WAIT') == 'WAIT']
        
        if tradeable:
            st.markdown(f"### 📊 {len(tradeable)} TRADEABLE SETUPS")
            
            for sig in tradeable:
                color = "#10b981" if sig['direction'] == 'LONG' else "#ef4444"
                time_ago = get_time_ago(sig['detected_at'])
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                            padding: 1rem; border-radius: 12px; margin-bottom: 1rem; color: white;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <h3 style="margin: 0;">{sig['exchange']} | {sig['symbol']}</h3>
                            <p style="margin: 0.3rem 0; font-size: 0.9rem;">{sig['timeframe']} | ${sig['price']:.4f}</p>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.3rem; font-weight: bold;">{sig['conviction']}</div>
                            <div style="background: rgba(0,0,0,0.3); padding: 0.3rem 0.6rem; border-radius: 6px; font-size: 0.85rem; margin-top: 0.3rem;">
                                ⏱️ {time_ago}
                            </div>
                        </div>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); padding: 0.5rem 0.8rem; border-radius: 8px; margin-top: 0.5rem;">
                        <strong>{sig['direction']}</strong> | Size: <strong>{sig['position_size']}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"**WHY:** {sig['reason']}")
                st.divider()
        
        if wait:
            with st.expander(f"⚪ {len(wait)} WAIT Signals", expanded=False):
                for sig in wait[:10]:
                    st.markdown(f"**{sig['symbol']} - {sig['timeframe']}**")
                    st.caption(sig['reason'])
                    st.markdown("---")
    else:
        st.info("👆 Tap SCAN")
        st.markdown("""
        ### 🧠 EXPANSION EDGE
        **Setups:** Squeeze OR Crossover (independent)  
        **Trade:** Only when EXPANSION starts  
        **Confirm:** Elephant bar OR Tail bar (required)  
        **Conviction:** HIGH A (1.0R), HIGH B (0.6-0.8R), CAUTION (0.3-0.5R), WAIT (0R)
        """)
    
    st.divider()
    st.caption("⚡ Expansion = Price moving away from SMA zone with intent | No elephant/tail bar = NO TRADE")

if __name__ == "__main__":
    main()
