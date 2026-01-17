import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import time

# Page config - Mobile optimized
st.set_page_config(
    page_title="MEXC Sniper",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for mobile browsers
st.markdown("""
<style>
    /* Mobile-first responsive design */
    .main {
        padding: 0.5rem !important;
        max-width: 100% !important;
    }
    
    /* Remove Streamlit branding for cleaner mobile view */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Optimize for mobile touch */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        font-weight: bold;
        font-size: 1rem;
        margin: 0.5rem 0;
        touch-action: manipulation;
    }
    
    /* Mobile-friendly metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }
    
    /* Signal cards optimized for mobile */
    .signal-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        touch-action: manipulation;
    }
    
    /* Responsive columns for mobile */
    @media (max-width: 768px) {
        .row-widget.stHorizontal {
            flex-direction: column !important;
        }
        
        div[data-testid="column"] {
            width: 100% !important;
            margin-bottom: 1rem;
        }
        
        .stButton>button {
            font-size: 1.1rem;
            height: 4em;
        }
    }
    
    /* Larger touch targets for mobile */
    .stSelectbox, .stMultiselect {
        font-size: 1rem;
    }
    
    /* Better spacing for mobile */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Sticky header for mobile scrolling */
    div[data-testid="stMetricDelta"] {
        display: none;
    }
    
    /* Improve readability on small screens */
    p, div, span, label {
        line-height: 1.6;
    }
    
    /* Hide sidebar toggle on mobile for cleaner UI */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = True

# Conditional auto-refresh based on user preference
if st.session_state.auto_refresh_enabled:
    st_autorefresh(interval=60000, key="datarefresh")

# Initialize MEXC exchange
@st.cache_resource
def init_exchange():
    try:
        exchange = ccxt.mexc({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'timeout': 30000
        })
        return exchange
    except Exception as e:
        st.error(f"❌ Failed to initialize MEXC: {e}")
        return None

# Fetch OHLCV data
@st.cache_data(ttl=60)
def fetch_ohlcv(exchange, symbol, timeframe, limit=200):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        return None

# Calculate indicators
def calculate_indicators(df):
    if df is None or len(df) < 200:
        return None
    
    # SMA
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['sma_200'] = ta.sma(df['close'], length=200)
    
    # RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # VWAP
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    
    # Average range for Elephant Bar detection
    df['range'] = df['high'] - df['low']
    df['avg_range'] = df['range'].rolling(20).mean()
    df['body'] = abs(df['close'] - df['open'])
    
    return df

# Detect patterns
def detect_patterns(df, symbol, timeframe):
    if df is None or len(df) < 200:
        return []
    
    signals = []
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Golden Cross
    if prev['sma_20'] <= prev['sma_200'] and latest['sma_20'] > latest['sma_200']:
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'pattern': '🌟 Golden Cross',
            'type': 'BULLISH',
            'price': latest['close'],
            'rsi': latest['rsi'],
            'vwap_diff': ((latest['close'] - latest['vwap']) / latest['vwap'] * 100),
            'reasoning': 'SMA 20 crossed above SMA 200. Strong bullish momentum building.'
        })
    
    # Death Cross
    if prev['sma_20'] >= prev['sma_200'] and latest['sma_20'] < latest['sma_200']:
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'pattern': '💀 Death Cross',
            'type': 'BEARISH',
            'price': latest['close'],
            'rsi': latest['rsi'],
            'vwap_diff': ((latest['close'] - latest['vwap']) / latest['vwap'] * 100),
            'reasoning': 'SMA 20 crossed below SMA 200. Bearish pressure increasing.'
        })
    
    # SMA Squeeze
    sma_diff = abs(latest['sma_20'] - latest['sma_200']) / latest['sma_200'] * 100
    if sma_diff < 0.3:
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'pattern': '🔥 SMA Squeeze',
            'type': 'NEUTRAL',
            'price': latest['close'],
            'rsi': latest['rsi'],
            'vwap_diff': ((latest['close'] - latest['vwap']) / latest['vwap'] * 100),
            'reasoning': f'SMAs converging within {sma_diff:.2f}%. Breakout imminent.'
        })
    
    # Kiss of Life
    if (prev['low'] <= prev['sma_200'] and latest['close'] > latest['sma_200'] and 
        latest['close'] > latest['open']):
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'pattern': '💚 Kiss of Life',
            'type': 'BULLISH',
            'price': latest['close'],
            'rsi': latest['rsi'],
            'vwap_diff': ((latest['close'] - latest['vwap']) / latest['vwap'] * 100),
            'reasoning': 'Price rejected at 200 SMA support. Bullish bounce expected.'
        })
    
    # Kiss of Death
    if (prev['high'] >= prev['sma_200'] and latest['close'] < latest['sma_200'] and 
        latest['close'] < latest['open']):
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'pattern': '🔴 Kiss of Death',
            'type': 'BEARISH',
            'price': latest['close'],
            'rsi': latest['rsi'],
            'vwap_diff': ((latest['close'] - latest['vwap']) / latest['vwap'] * 100),
            'reasoning': 'Price rejected at 200 SMA resistance. Bearish reversal likely.'
        })
    
    # Divergence
    prev_diff = abs(prev['sma_20'] - prev['sma_200'])
    curr_diff = abs(latest['sma_20'] - latest['sma_200'])
    if curr_diff > prev_diff * 1.5 and sma_diff > 1.0:
        direction = 'bullish' if latest['sma_20'] > latest['sma_200'] else 'bearish'
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'pattern': '📊 Divergence',
            'type': 'NEUTRAL',
            'price': latest['close'],
            'rsi': latest['rsi'],
            'vwap_diff': ((latest['close'] - latest['vwap']) / latest['vwap'] * 100),
            'reasoning': f'SMA 20 pulling away from SMA 200. {direction.capitalize()} trend accelerating.'
        })
    
    # Elephant Bar
    if (latest['range'] > latest['avg_range'] * 2.5 and 
        latest['body'] > latest['range'] * 0.75):
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'pattern': '🐘 Elephant Bar',
            'type': 'STRONG',
            'price': latest['close'],
            'rsi': latest['rsi'],
            'vwap_diff': ((latest['close'] - latest['vwap']) / latest['vwap'] * 100),
            'reasoning': 'Massive volume bar detected (>2.5x avg). Strong directional move.'
        })
    
    return signals

# Check liquidity holes
def check_liquidity_hole(exchange, symbol):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        ohlcv_1m = exchange.fetch_ohlcv(symbol, '1m', limit=20)
        avg_volume_1m = np.mean([candle[5] for candle in ohlcv_1m])
        
        mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
        threshold = mid_price * 0.005
        
        bid_depth = sum([bid[1] for bid in orderbook['bids'] if bid[0] >= mid_price - threshold])
        ask_depth = sum([ask[1] for ask in orderbook['asks'] if ask[0] <= mid_price + threshold])
        total_depth = bid_depth + ask_depth
        
        if total_depth < avg_volume_1m * 0.2:
            return True, total_depth, avg_volume_1m
        
        return False, total_depth, avg_volume_1m
    except:
        return False, 0, 0

# Main scanning function
def scan_markets(exchange, symbols, timeframes):
    all_signals = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_scans = len(symbols) * len(timeframes)
    current_scan = 0
    
    for symbol in symbols:
        for timeframe in timeframes:
            current_scan += 1
            progress_bar.progress(current_scan / total_scans)
            status_text.text(f"📡 Scanning {symbol} on {timeframe}...")
            
            df = fetch_ohlcv(exchange, symbol, timeframe)
            if df is not None:
                df = calculate_indicators(df)
                signals = detect_patterns(df, symbol, timeframe)
                all_signals.extend(signals)
            
            time.sleep(0.1)
        
        # Check liquidity holes
        has_hole, depth, avg_vol = check_liquidity_hole(exchange, symbol)
        if has_hole:
            all_signals.append({
                'symbol': symbol,
                'timeframe': 'ALL',
                'pattern': '🕳️ Liquidity Hole',
                'type': 'WARNING',
                'price': 0,
                'rsi': 0,
                'vwap_diff': 0,
                'reasoning': f'Thin order book. Depth: {depth:.2f}, Avg Vol: {avg_vol:.2f}. High volatility expected.'
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return all_signals

# Main app
def main():
    # Header - Mobile optimized
    st.markdown("# 🎯 MEXC Prop Sniper")
    st.caption("📱 Real-time market scanner optimized for mobile")
    
    exchange = init_exchange()
    if exchange is None:
        st.error("❌ Failed to connect to MEXC. Please check your connection.")
        st.stop()
    
    # Mobile-friendly expandable settings
    with st.expander("⚙️ SETTINGS", expanded=False):
        # Fetch available USDT pairs
        try:
            markets = exchange.load_markets()
            usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT') and markets[symbol]['active']]
            usdt_pairs.sort()
        except:
            usdt_pairs = []
            st.error("Failed to load markets")
        
        st.markdown("### 📊 Trading Pairs")
        selected_pairs = st.multiselect(
            "Select pairs to scan",
            usdt_pairs[:50],
            default=usdt_pairs[:10] if usdt_pairs else [],
            help="Choose which trading pairs to monitor"
        )
        
        st.markdown("### ⏱️ Timeframes")
        selected_timeframes = st.multiselect(
            "Select timeframes",
            ['3m', '5m', '15m', '1h', '4h'],
            default=['5m', '15m', '1h'],
            help="Choose which timeframes to analyze"
        )
        
        st.markdown("### 🔄 Auto-Refresh")
        auto_refresh = st.toggle(
            "Enable auto-refresh (60s)",
            value=st.session_state.auto_refresh_enabled,
            help="Automatically scan markets every 60 seconds"
        )
        st.session_state.auto_refresh_enabled = auto_refresh
    
    # Large scan button for mobile
    scan_button = st.button("🔍 SCAN MARKETS NOW", type="primary", use_container_width=True)
    
    # Metrics row - Mobile responsive
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Signals", len(st.session_state.signals))
    with col2:
        st.metric("📡 Exchange", "MEXC")
    with col3:
        if st.session_state.last_update:
            update_time = st.session_state.last_update.strftime("%H:%M")
            st.metric("🕐 Updated", update_time)
        else:
            st.metric("🕐 Updated", "Never")
    
    st.divider()
    
    # Scan markets
    if scan_button or (not st.session_state.signals and selected_pairs):
        if not selected_pairs:
            st.warning("⚠️ Please select at least one trading pair in settings")
        elif not selected_timeframes:
            st.warning("⚠️ Please select at least one timeframe in settings")
        else:
            with st.spinner("🔍 Scanning markets..."):
                signals = scan_markets(exchange, selected_pairs, selected_timeframes)
                st.session_state.signals = signals
                st.session_state.last_update = datetime.now()
                st.rerun()
    
    # Display signals - Mobile optimized
    if st.session_state.signals:
        st.markdown(f"### 📊 {len(st.session_state.signals)} Active Signals")
        
        for idx, signal in enumerate(st.session_state.signals):
            # Determine card color
            if signal['type'] == 'BULLISH':
                card_color = "#10b981"
                card_gradient = "linear-gradient(135deg, #10b981 0%, #059669 100%)"
            elif signal['type'] == 'BEARISH':
                card_color = "#ef4444"
                card_gradient = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
            elif signal['type'] == 'STRONG':
                card_color = "#8b5cf6"
                card_gradient = "linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)"
            elif signal['type'] == 'WARNING':
                card_color = "#f59e0b"
                card_gradient = "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
            else:
                card_color = "#3b82f6"
                card_gradient = "linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)"
            
            # Signal card
            st.markdown(f"""
            <div style="background: {card_gradient}; 
                        padding: 1.5rem; 
                        border-radius: 15px; 
                        margin-bottom: 1.5rem; 
                        color: white;
                        box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                    <div>
                        <h2 style="margin: 0; font-size: 1.5rem;">{signal['symbol']}</h2>
                        <p style="margin: 0.5rem 0; font-size: 1.1rem; opacity: 0.95;">{signal['pattern']}</p>
                        <span style="background: rgba(255,255,255,0.2); 
                                     padding: 0.3rem 0.8rem; 
                                     border-radius: 20px; 
                                     font-size: 0.85rem;
                                     display: inline-block;">
                            {signal['timeframe']}
                        </span>
                    </div>
                    <div style="text-align: right;">
                        <p style="margin: 0; font-weight: bold; font-size: 1.3rem;">
                            ${signal['price']:.4f}
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Indicators - Mobile friendly
            col1, col2 = st.columns(2)
            with col1:
                rsi_status = "🔴 Overbought" if signal['rsi'] > 70 else "🟢 Oversold" if signal['rsi'] < 30 else "🟡 Neutral"
                st.info(f"**RSI:** {signal['rsi']:.1f}\n\n{rsi_status}")
            with col2:
                vwap_status = "✅ Above" if signal['vwap_diff'] > 0 else "⚠️ Below"
                st.info(f"**VWAP:** {signal['vwap_diff']:.2f}%\n\n{vwap_status}")
            
            # Reasoning
            st.success(f"**💡 Analysis:** {signal['reasoning']}")
            
            if idx < len(st.session_state.signals) - 1:
                st.divider()
    else:
        # Empty state - Mobile friendly
        st.info("👆 Tap **SCAN MARKETS NOW** to find trading signals")
        st.markdown("""
        ### 🚀 Quick Start
        1. Open **⚙️ SETTINGS** above
        2. Select your favorite trading pairs
        3. Choose timeframes to scan
        4. Tap the scan button
        
        💡 **Tip:** Enable auto-refresh to get updates every 60 seconds!
        """)
    
    # Footer
    st.divider()
    st.caption("📊 Timeframes: 3m • 5m • 15m • 1h • 4h")
    st.caption("📈 Indicators: SMA 20/200 • RSI • VWAP")
    st.caption("🔄 Auto-refresh updates every 60 seconds when enabled")

if __name__ == "__main__":
    main()
