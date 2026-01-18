import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import time

# --- CONFIGURATION ---
BATCH_SIZE = 5  # Number of pairs to scan before a rest
REST_TIME = 1.5 # Seconds to rest between batches to avoid rate limits

st.set_page_config(page_title="MEXC Pro Scanner", page_icon="🎯", layout="wide")

# (Keep your existing CSS here)

@st.cache_resource
def init_exchange():
    return ccxt.mexc({'enableRateLimit': True, 'options': {'defaultType': 'spot'}, 'timeout': 30000})

@st.cache_data(ttl=600)
def get_mexc_pairs(_exchange):
    try:
        markets = _exchange.load_markets()
        # Filter for USDT pairs and ensure BTC/USDT is included
        pairs = [s for s in markets.keys() if s.endswith('/USDT') and markets[s]['active']]
        # Move BTC/USDT to the front of the list
        if 'BTC/USDT' in pairs:
            pairs.insert(0, pairs.pop(pairs.index('BTC/USDT')))
        return pairs
    except:
        return ['BTC/USDT', 'ETH/USDT']

def scan_markets(exchange, symbols, timeframes):
    all_signals = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_symbols = len(symbols)
    
    for i in range(0, total_symbols, BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        
        for symbol in batch:
            # Update progress based on symbol count
            progress = (symbols.index(symbol) + 1) / total_symbols
            progress_bar.progress(progress)
            status_text.text(f"📡 Batch {i//BATCH_SIZE + 1}: Scanning {symbol}...")
            
            liquidity_data = check_liquidity(exchange, symbol)
            
            for timeframe in timeframes:
                try:
                    df = fetch_ohlcv(exchange, symbol, timeframe)
                    if df is not None:
                        df = calculate_indicators(df)
                        if df is not None:
                            signals = detect_signals(df, symbol, timeframe, liquidity_data)
                            all_signals.extend(signals)
                except Exception:
                    continue
        
        # REST after each batch (except the last one)
        if i + BATCH_SIZE < total_symbols:
            status_text.text(f"⏳ Cooling down to respect MEXC limits...")
            time.sleep(REST_TIME)
    
    progress_bar.empty()
    status_text.empty()
    return all_signals

def main():
    st.markdown("# 🎯 MEXC Pro Scanner")
    exchange = init_exchange()
    
    usdt_pairs = get_mexc_pairs(exchange)
    
    with st.expander("⚙️ SETTINGS", expanded=False):
        # Increased limit to 100 for your request, with BTC as default
        selected_pairs = st.multiselect(
            "Select pairs", 
            usdt_pairs[:100], 
            default=['BTC/USDT'] + usdt_pairs[1:10] # BTC + next 9
        )
        
        # Using your preferred timeframes from saved info
        selected_timeframes = st.multiselect(
            "Select timeframes", 
            ['3m', '5m', '15m', '1h', '4h'], 
            default=['15m', '1h', '4h']
        )
        
        auto_refresh = st.toggle("Enable (60s)", value=st.session_state.get('auto_refresh_enabled', True))
        st.session_state.auto_refresh_enabled = auto_refresh

    # (Keep the rest of your UI and signal display logic here)

if __name__ == "__main__":
    main()
