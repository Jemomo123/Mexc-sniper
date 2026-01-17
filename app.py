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
def fetch_ohlcv(exchange, symbol, timeframe, limit=200):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        return None

# Calculate indicators
def calculate_indicators(df):
    if df is None or len(df) < 200:
        return None
    
    try:
        # SMA
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_200'] = ta.sma(df['close'], length=200)
        
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # VWAP - Now with proper DatetimeIndex
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # Average range for Elephant Bar detection
        df['range'] = df['high'] - df['low']
        df['avg_range'] = df['range'].rolling(20).mean()
        df['body'] = abs(df['close'] - df['open'])
        
        # Fill NaN values for VWAP if needed
        if df['vwap'].isna().all():
            df['vwap'] = df['close']
        else:
            df['vwap'].fillna(method='ffill', inplace=True)
            df['vwap'].fillna(method='bfill', inplace=True)
        
        return df
    except Exception as e:
        return None

# Detect patterns
def detect_patterns(df, symbol, timeframe):
    if df is None or len(df) < 200:
        return []
    
    signals = []
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Check for None values
    if pd.isna(latest['sma_20']) or pd.isna(latest['sma_200']) or pd.isna(latest['rsi']) or pd.isna(latest['vwap']):
        return []
    
    # Golden Cross
    if not pd.isna(prev['sma_20']) and not pd.isna(prev['sma_200']):
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
    if not pd.isna(prev['sma_20']) and not pd.isna(prev['sma_200']):
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
    if not pd.isna(prev['sma_200']):
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
    if not pd.isna(prev['sma_200']):
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
    if not pd.isna(prev['sma_20']) and not pd.isna(prev['sma_200']):
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
    if not pd.isna(latest['avg_range']):
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
            
            try:
                df = fetch_ohlcv(exchange, symbol, timeframe)
                if df is not None:
                    df = calculate_indicators(df)
                    if df is not None:
                        signals = detect_patterns(df, symbol, timeframe)
                        all_signals.extend(signals)
            except Exception as e:
                continue
            
            time.sleep(0.1)
        
        # Check liquidity holes
        try:
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
        except:
            continue
    
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
    with st.exp
