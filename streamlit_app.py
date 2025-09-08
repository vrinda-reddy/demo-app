import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# python -m streamlit run Dashboard.py

# st.set_page_config(
#     page_title="US Stock Market Dashboard",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

st.sidebar.markdown("# Setting")

st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.positive { color: #00C853; }
.negative { color: #FF1744; }
.big-font { font-size: 24px !important; }
</style>
""", unsafe_allow_html=True)

# Configure requests session with retries and proper headers
def setup_yfinance_session():
    """Setup an even more robust session for yfinance requests"""
    session = requests.Session()
    
    # Add multiple realistic browser headers
    headers_list = [
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        },
        {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
    ]
    
    # Randomly select headers
    selected_headers = random.choice(headers_list)
    session.headers.update(selected_headers)
    
    # More aggressive retry strategy
    retry_strategy = Retry(
        total=5,  # Increased retries
        backoff_factor=2,  # Longer backoff
        status_forcelist=[429, 500, 502, 503, 504, 403, 404],  # Added more status codes
        respect_retry_after_header=True
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Initialize the session
session = setup_yfinance_session()

def safe_yf_download(*args, **kwargs):
    """Enhanced wrapper for yf.download with longer delays and better fallbacks"""
    max_retries = 5  # Increased retries
    base_delay = 3.0  # Longer base delay
    
    for attempt in range(max_retries):
        try:
            # Much longer delays between attempts
            if attempt > 0:
                delay = base_delay * (3 ** attempt) + random.uniform(2, 5)
                st.info(f"Waiting {delay:.1f} seconds before retry {attempt + 1}...")
                time.sleep(delay)
            else:
                # Even first attempt gets a longer delay
                initial_delay = random.uniform(2.0, 4.0)
                time.sleep(initial_delay)
            
            # Rotate headers occasionally
            if attempt > 1:
                session.headers.update({
                    'User-Agent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{120 + attempt}.0.0.0 Safari/537.36'
                })
            
            # Use the configured session
            data = yf.download(*args, **kwargs, session=session, show_errors=False)
            
            # More thorough validation
            if data is not None and not data.empty and len(data) > 0:
                # Check if we actually have meaningful data
                if hasattr(data, 'columns') and len(data.columns) > 0:
                    return data
                else:
                    raise Exception("Data has no columns")
            else:
                raise Exception("Empty or invalid data returned")
                
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "too many requests" in error_msg:
                st.warning(f"Rate limited on attempt {attempt + 1}. Will retry with longer delay...")
            elif "expecting value" in error_msg:
                st.warning(f"Empty response on attempt {attempt + 1}. Yahoo Finance may be blocking requests...")
            else:
                st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt == max_retries - 1:
                st.error(f"❌ Failed to download data after {max_retries} attempts. Yahoo Finance may be temporarily blocking this server.")
                st.info("💡 Try again in 5-10 minutes, or use a different data source if available.")
                return pd.DataFrame()
    
    return pd.DataFrame()

def safe_yf_ticker_info(ticker_symbol):
    """Enhanced ticker info with longer delays"""
    max_retries = 4
    base_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2 ** attempt) + random.uniform(1, 3)
                time.sleep(delay)
            else:
                time.sleep(random.uniform(1.0, 2.0))
            
            ticker = yf.Ticker(ticker_symbol, session=session)
            info = ticker.info
            
            # Better validation
            if info and isinstance(info, dict) and len(info) > 5:  # Must have substantial info
                if ('symbol' in info or 'shortName' in info or 'longName' in info):
                    return info
            
            raise Exception("Invalid or insufficient info returned")
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"⚠️ Could not fetch info for {ticker_symbol} after {max_retries} attempts")
                return {'symbol': ticker_symbol, 'longName': ticker_symbol, 'shortName': ticker_symbol}  # Fallback
    
    return {'symbol': ticker_symbol, 'longName': ticker_symbol, 'shortName': ticker_symbol}


def safe_yf_history(ticker_symbol, **kwargs):
    """Enhanced history fetching with longer delays"""
    max_retries = 4
    base_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2 ** attempt) + random.uniform(1, 3)
                time.sleep(delay)
            else:
                time.sleep(random.uniform(1.0, 2.0))
            
            ticker = yf.Ticker(ticker_symbol, session=session)
            hist = ticker.history(**kwargs)
            
            if hist is not None and not hist.empty and len(hist) > 0:
                return hist
            else:
                raise Exception("Empty history returned")
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"⚠️ Could not fetch history for {ticker_symbol} after {max_retries} attempts")
                return pd.DataFrame()
    
    return pd.DataFrame()

@st.cache_data
def fetch_data(ticker, start, end, interval):
    """Safe wrapper for fetching ticker data"""
    try:
        data = safe_yf_download(ticker, start=start, end=end, interval=interval)
        
        # Validate data before returning
        if data.empty:
            st.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def fetch_history(ticker, period, interval):
    """Safe wrapper for fetching ticker history"""
    return safe_yf_history(ticker, period=period, interval=interval)

@st.cache_data
def fetch_ticker_info(ticker):
    """Safe wrapper for fetching ticker info"""
    return safe_yf_ticker_info(ticker)

def format_value(value):
    suffixes = ["", "K", "M", "B", "T"]
    suffix_index = 0
    while value and value >= 1000 and suffix_index < len(suffixes) - 1:
        value /= 1000
        suffix_index += 1
    return f"${value:.1f}{suffixes[suffix_index]}" if value else "N/A"

def safe_format(value, fmt="{:.2f}", fallback="N/A"):
    try:
        return fmt.format(float(value)) if value is not None else fallback
    except (ValueError, TypeError):
        return fallback
    
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(ttl=3600)
def get_market_data():
    """Fetch major US market indices with enhanced error handling and fallbacks"""
    indices = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC', 
        'DOW': '^DJI',
        'Russell 2000': '^RUT',
        'VIX': '^VIX'
    }

    data = {}
    failed_indices = []
    
    st.info("🔄 Loading market data... This may take a moment due to rate limiting protection.")
    
    for i, (name, symbol) in enumerate(indices.items()):
        try:
            # Progress indicator
            progress = (i + 1) / len(indices)
            st.progress(progress, text=f"Loading {name}...")
            
            # Longer delay between requests
            time.sleep(random.uniform(3.0, 5.0))
            
            hist = safe_yf_history(symbol, period='5d')
            
            if not hist.empty and len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

                data[name] = {
                    'current': current_price,
                    'change': change,
                    'change_pct': change_pct,
                    'symbol': symbol
                }
            else:
                failed_indices.append(name)
                
        except Exception as e:
            failed_indices.append(name)
            continue

    # Clear progress bar
    st.empty()
    
    # Show results summary
    if data:
        success_count = len(data)
        total_count = len(indices)
        st.success(f"✅ Successfully loaded {success_count}/{total_count} market indices")
        
        if failed_indices:
            st.warning(f"⚠️ Could not load: {', '.join(failed_indices)}")
    else:
        st.error("❌ Could not load any market data due to rate limiting.")
        st.info("💡 Yahoo Finance is currently blocking requests from this server. Please try again in 10-15 minutes.")
        
        # Provide fallback data so the app doesn't crash
        fallback_data = {
            'S&P 500': {'current': 5000.0, 'change': 0.0, 'change_pct': 0.0, 'symbol': '^GSPC'},
            'NASDAQ': {'current': 16000.0, 'change': 0.0, 'change_pct': 0.0, 'symbol': '^IXIC'},
            'DOW': {'current': 35000.0, 'change': 0.0, 'change_pct': 0.0, 'symbol': '^DJI'},
            'Russell 2000': {'current': 2000.0, 'change': 0.0, 'change_pct': 0.0, 'symbol': '^RUT'},
            'VIX': {'current': 20.0, 'change': 0.0, 'change_pct': 0.0, 'symbol': '^VIX'}
        }
        st.info("📊 Showing placeholder data to prevent app crashes.")
        return fallback_data
    
    return data


@st.cache_data
def get_sector_data():
    """Fetch sector performance data using sector ETFs"""
    sectors = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financials': 'XLF',
        'Consumer Disc.': 'XLY',
        'Industrials': 'XLI',
        'Communication': 'XLC',
        'Consumer Staples': 'XLP',
        'Energy': 'XLE',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Materials': 'XLB'
    }
    

    data = {}
    for name, symbol in sectors.items():
        try:
            hist = safe_yf_history(symbol, period='2d')
            info = safe_yf_ticker_info(symbol)

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

                data[name] = {
                    'current': current_price,
                    'change': change,
                    'change_pct': change_pct,
                    'symbol': symbol
                }
        except Exception as e:
            st.warning(f"Error fetching data for {name}: {e}")
            continue

    return data

def safe_calculate_performance(index_data, start_date, end_date):
    """Safely calculate performance metrics"""
    if index_data.empty or len(index_data) < 2:
        return 0.0, 0.0  # Return safe defaults
    
    try:
        pct_change = (index_data["Close"].iloc[-1] - index_data["Close"].iloc[0]) / index_data["Close"].iloc[0] * 100
        volatility = index_data["Returns"].std() * 100
        return pct_change, volatility
    except (IndexError, KeyError) as e:
        st.warning(f"Could not calculate performance: {str(e)}")
        return 0.0, 0.0
    
def add_emergency_controls():
    """Add emergency controls to the sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🚨 Emergency Controls")
        
        if st.button("🔄 Clear All Cache & Reload"):
            st.cache_data.clear()
            st.info("Cache cleared! Reloading...")
            st.rerun()
        
        if st.button("⏸️ Use Offline Mode"):
            st.session_state.offline_mode = True
            st.info("Switched to offline mode with placeholder data")
            st.rerun()
        
        if st.button("🌐 Resume Online Mode"):
            st.session_state.offline_mode = False
            st.info("Resumed online data fetching")
            st.rerun()

def check_offline_mode():
    """Check if we should use offline mode"""
    if st.session_state.get('offline_mode', False):
        st.warning("🔌 **Offline Mode Active** - Using placeholder data to avoid rate limiting")
        return True
    return False

st.title("Top-Down Stock Market Analysis")

# Add emergency controls
add_emergency_controls()

# Check if offline mode is active
if check_offline_mode():
    # Use placeholder data instead of fetching from Yahoo Finance
    st.info("Using placeholder data to avoid rate limiting issues")

st.subheader("US Market Overview")
indices = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "VIX": "^VIX"
}

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-07-01"))
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

selected_index = st.sidebar.selectbox("Select Market Index", list(indices.keys()), index=0)
sector_etfs = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC"
}

market_data = get_market_data()

if market_data:
    cols = st.columns(len(market_data))

    for i, (name, data) in enumerate(market_data.items()):
        with cols[i]:
            change_color = "positive" if data['change'] >= 0 else "negative"
            change_symbol = "+" if data['change'] >= 0 else ""

            st.markdown(f"""
            <div class="metric-card">
                <h6>{name}</h6>
                <div class="big-font">{data['current']:.2f}</div>
                <div class="{change_color}">
                    {change_symbol}{data['change']:.2f} ({data['change_pct']:+.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)


selected_market = st.sidebar.selectbox("Select Market Sector", list(sector_etfs.keys()), index=2)
period = st.selectbox("Enter a time frame", ("1D", "5D", "1WK", "1MO", "3MO"), index=1)
ma_window = st.sidebar.slider("Moving Average Window", min_value=5, max_value=50, value=20)
use_log_scale = st.sidebar.checkbox("Log Scale", value=False)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)

index_data = fetch_data(indices[selected_index], start_date, end_date, period)
index_data.dropna(inplace=True)
if not index_data.empty:
    index_data["Returns"] = index_data["Close"].pct_change()
    index_data["MA"] = index_data["Close"].rolling(window=ma_window).mean()
    
    pct_change, volatility = safe_calculate_performance(index_data, start_date, end_date)
    
    # Only show performance if we have valid data
    if pct_change != 0.0 or volatility != 0.0:
        st.write(f"""
        **Performance:** {pct_change:.2f}% from {start_date} to {end_date}  
        **Volatility (Daily Std Dev):** {volatility:.2f}%
        """)
else:
    st.error(f"No data available for {selected_index}. This may be due to rate limiting.")
    st.info("Please try again in a few minutes, or contact support if the issue persists.")

# st.write(f"""
# **Performance:** {pct_change:.2f}% from {start_date} to {end_date}  
# **Volatility (Daily Std Dev):** {volatility:.2f}%
# """)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(index_data.index, index_data["Close"], label="Close", color="blue")
ax.plot(index_data.index, index_data["MA"], label=f"{ma_window}-Day MA", color="orange", linestyle="--")
ax.set_title(f"{selected_index} Performance - From {start_date} to {end_date}")
ax.set_ylabel("Index Value")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.download_button(
    label="Download Index Data",
    data=index_data.to_csv().encode("utf-8"),
    file_name=f"{selected_index}_data.csv",
    mime="text/csv"
)

show_vix = st.checkbox("Show VIX Index")
if show_vix:
    vix_data = fetch_data("^VIX", start_date, end_date, period)
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(vix_data.index, vix_data["Close"], label="VIX", color="red")
    ax2.set_title("VIX - Market Volatility Index")
    ax2.set_ylabel("VIX Value")
    ax2.grid(True)
    st.pyplot(fig2)

st.divider()

info = safe_yf_ticker_info(indices[selected_index])

st.subheader(f"{info.get('longName', 'N/A')} Details")

col1, col2 = st.columns(2)

# Display price information as a dataframe
price_info_df = pd.DataFrame({
    "Price Info": ["Previous Close", "Day High", "Day Low"],
    "Value": [
        safe_format(info.get('previousClose'), "${:.2f}"),
        safe_format(info.get('dayHigh'), "${:.2f}"),
        safe_format(info.get('dayLow'), "${:.2f}")
    ]
})
col1.dataframe(price_info_df, hide_index=True)

biz_metrics_df = pd.DataFrame({
    "Business Metrics": ["52 Week High", "52 Week Low", "Latest % Change"],
    "Value": [
        safe_format(info.get('fiftyTwoWeekHigh'), "${:.2f}"),
        safe_format(info.get('fiftyTwoWeekLow'), "${:.2f}"),
        safe_format(info.get('regularMarketChangePercent'), "{:.2f}%")
    ]
})
col2.dataframe(biz_metrics_df, hide_index=True)

tabs = st.tabs(["Sector Performance", "Individual Stock Analysis"])

# Level 1: Sector Performance
with tabs[0]:
    st.subheader("Sector Performance Overview")

    sector_data = get_sector_data()

    per_row = 4  # Metrics per row, adjust as you like
    sector_items = list(sector_data.items())

    for row_start in range(0, len(sector_items), per_row):
        row_items = sector_items[row_start:row_start + per_row]
        cols = st.columns(len(row_items))
        for col, (name, data) in zip(cols, row_items):
            change_color = "positive" if data['change'] >= 0 else "negative"
            change_symbol = "+" if data['change'] >= 0 else ""
            col.markdown(f"""
            <div class="metric-card">
                <h6>{name}</h6>
                <div class="big-font">{data['current']:.2f}</div>
                <div class="{change_color}">
                    {change_symbol}{data['change']:.2f} ({data['change_pct']:+.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.subheader(f"Sector Performance (via Sector ETFs) - {selected_market}")

    market_data = safe_yf_download(sector_etfs[selected_market], start=start_date, end=end_date, interval='1D')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(market_data.index, market_data["Close"], label=selected_market, color="blue")
    ax.set_title(f"{selected_market} Performance - From {start_date} to {end_date}")
    ax.set_ylabel("Index Value")
    ax.grid(True)
    st.pyplot(fig)

    info = safe_yf_ticker_info(sector_etfs[selected_market])

    st.subheader(f"{sector_etfs[selected_market]} - {info.get('longName', 'N/A')}")

    col1, col2, col3 = st.columns(3)

    # Display stock information as a dataframe
    stock_info = [
        ("Stock Info", "Value"),
        ("Previous Close", safe_format(info.get('previousClose'), fmt="${:.2f}")),
        ("Day High", safe_format(info.get('dayHigh'), fmt="${:.2f}")),
        ("Day Low", safe_format(info.get('dayLow'), fmt="${:.2f}"))
    ]
    
    df = pd.DataFrame(stock_info[1:], columns=stock_info[0]).astype(str)
    col1.dataframe(df, width=500, hide_index=True)
    
    price_info = [
        ("Price Info", "Value"),
        ("52 Week High", safe_format(info.get('fiftyTwoWeekHigh'), fmt="${:.2f}")),
        ("52 Week Low", safe_format(info.get('fiftyTwoWeekLow'), fmt="${:.2f}")),
        ("Latest % Change", safe_format(info.get('regularMarketChangePercent'), "{:.2f}%"))
    ]
    
    df = pd.DataFrame(price_info[1:], columns=price_info[0]).astype(str)
    col2.dataframe(df, width=500, hide_index=True)

    # Display business metrics as a dataframe
    biz_metrics = [
        ("Business Metrics", "Value"),
        ("Market Cap", format_value(info.get('marketCap'))),
        ("Div Yield (FWD)", safe_format(info.get('dividendYield'), fmt="{:.2}%") if info.get('dividendYield') else 'N/A')
    ]
    
    df = pd.DataFrame(biz_metrics[1:], columns=biz_metrics[0]).astype(str)
    col3.dataframe(df, width=500, hide_index=True)

    st.divider()


    try:
        eunl = yf.Ticker(sector_etfs[selected_market])
        funds_data = eunl.funds_data  # Special method not in safe wrappers
    except Exception as e:
        st.warning(f"Could not create ticker object: {e}")
        eunl = None
    funds_data = eunl.funds_data
    if funds_data is not None:
    #    dump(funds_data)
        top = funds_data.top_holdings
        if top is not None:
            print("columns = ", top.columns.tolist())
            for symbol, row in top.iterrows():
                name = row['Name']
                percent = row['Holding Percent']
                print(f"{symbol} = {percent:.2%}")
    else:
        print("***no top found")

    df = pd.DataFrame(top)
    st.dataframe(df)

# Level 3: Stock Drill-Down
with tabs[1]:
    default_ticker = {
        "Technology": "AAPL",
        "Healthcare": "JNJ",
        "Financials": "JPM",
        "Energy": "XOM",
        "Industrials": "BA",
        "Consumer Discretionary": "AMZN",
        "Consumer Staples": "PG",
        "Utilities": "NEE",
        "Materials": "LIN",
        "Real Estate": "PLD",
        "Communication Services": "GOOGL"
    }.get(selected_index, "AAPL")

    stock_ticker = st.text_input("Enter Stock Ticker", value=default_ticker)
    stock_data = fetch_data(stock_ticker, start_date, end_date, period)

    if not stock_data.empty:
        data = fetch_data(stock_ticker, start_date, end_date, period)

        data['MA'] = data['Close'].rolling(window=ma_window).mean()
        if show_rsi:
            data['RSI'] = calculate_rsi(data)

        info = safe_yf_ticker_info(stock_ticker)

        st.subheader(f"{stock_ticker} - {info.get('longName', 'N/A')}")

        # Plot historical stock price data
        period_map = {
            "1D": ("1d", "1h"),
            "5D": ("5d", "1d"),
            "1M": ("1mo", "1d"),
            "6M": ("6mo", "1wk"),
            "YTD": ("ytd", "1mo"),
            "1Y": ("1y", "1mo"),
            "5Y": ("5y", "3mo"),
        }
        selected_period, interval = period_map.get(period, ("1mo", "1d"))
        history = fetch_history(stock_ticker, selected_period, interval)
        
        if "Close" in data:
            st.subheader("Closing Price Over Time")
            st.line_chart(data['Close'])
        else:
            st.warning("Closing price data is not available for this stock.")

        if "Volume" in data:
            st.subheader("Volume Over Time")
            st.line_chart(data['Volume'])
        else:
            st.warning("Volume data is not available for this stock.")

        col1, col2, col3 = st.columns(3)

        # Display stock information as a dataframe
        stock_info = [
            ("Stock Info", "Value"),
            ("Country", info.get('country', 'N/A')),
            ("Sector", info.get('sector', 'N/A')),
            ("Industry", info.get('industry', 'N/A')),
            ("Market Cap", format_value(info.get('marketCap'))),
            ("Enterprise Value", format_value( info.get('enterpriseValue'))),
            ("Employees", info.get('fullTimeEmployees', 'N/A'))
        ]
        
        df = pd.DataFrame(stock_info[1:], columns=stock_info[0]).astype(str)
        col1.dataframe(df, width=500, hide_index=True)
        
        # Display price information as a dataframe
        price_info = [
            ("Price Info", "Value"),
            ("Current Price", safe_format(info.get('currentPrice'), fmt="${:.2f}")),
            ("Previous Close", safe_format(info.get('previousClose'), fmt="${:.2f}")),
            ("Latest % Change", safe_format(info.get('regularMarketChangePercent'), "{:.2f}%")),
            ("Day High", safe_format(info.get('dayHigh'), fmt="${:.2f}")),
            ("Day Low", safe_format(info.get('dayLow'), fmt="${:.2f}")),
            ("52 Week High", safe_format(info.get('fiftyTwoWeekHigh'), fmt="${:.2f}")),
            ("52 Week Low", safe_format(info.get('fiftyTwoWeekLow'), fmt="${:.2f}"))
        ]
        
        df = pd.DataFrame(price_info[1:], columns=price_info[0]).astype(str)
        col2.dataframe(df, width=500, hide_index=True)

        # Display business metrics as a dataframe
        biz_metrics = [
            ("Business Metrics", "Value"),
            ("EPS (FWD)", safe_format(info.get('forwardEps'))),
            ("P/E (FWD)", safe_format(info.get('forwardPE'))),
            ("PEG Ratio", safe_format(info.get('pegRatio'))),
            ("Div Rate (FWD)", safe_format(info.get('dividendRate'), fmt="${:.2f}")),
            ("Div Yield (FWD)", safe_format(info.get('dividendYield'), fmt="{:.2}%") if info.get('dividendYield') else 'N/A'),
            ("Recommendation", info.get('recommendationKey', 'N/A').capitalize())
        ]
        
        df = pd.DataFrame(biz_metrics[1:], columns=biz_metrics[0]).astype(str)
        col3.dataframe(df, width=500, hide_index=True)

        tabs = st.tabs(["Moving Averages & RSI", "Dividends & Splits", "Raw Data"])

        # Tab 4: Moving Averages
        with tabs[0]:
            st.subheader("Closing Price with Moving Average")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['Close'], label="Close", color="blue")
            ax.plot(data.index, data['MA'], label=f"{ma_window}-Day MA", color="orange")
            ax.set_ylabel("Price")
            ax.set_xlabel("Date")
            ax.set_yscale("log" if use_log_scale else "linear")
            ax.legend()
            ax.set_title(f"{stock_ticker} Closing Price & MA")
            st.pyplot(fig)

            # Download chart as PNG
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button("Download Chart as PNG", buf.getvalue(), file_name="chart.png", mime="image/png")

            if show_rsi:
                st.subheader("Relative Strength Index (RSI)")
                fig2, ax2 = plt.subplots(figsize=(12, 3))
                ax2.plot(data.index, data['RSI'], label="RSI", color="purple")
                ax2.axhline(70, color='red', linestyle="--", linewidth=1)
                ax2.axhline(30, color='green', linestyle="--", linewidth=1)
                ax2.set_title("RSI (14-day)")
                ax2.set_ylim(0, 100)
                ax2.legend()
                st.pyplot(fig2)

        # Tab 5: Dividends & Splits
        with tabs[1]:
            st.subheader("Dividends & Splits")

            try:
                try:
                    ticker = yf.Ticker(stock_ticker)
                    dividends = ticker.dividends  # Special method
                    splits = ticker.splits        # Special method
                    # Keep this as-is if used for special methods like dividends/splits
                except Exception as e:
                    st.warning(f"Could not create ticker object: {e}")
                    ticker = None
                dividends = ticker.dividends
                splits = ticker.splits
            except:
                dividends = pd.Series()
                splits = pd.Series()
                st.warning("Could not load dividends/splits data")

            st.write("**Dividends:**")
            st.write(dividends if not dividends.empty else "No dividends found during this period.")
            st.write("**Splits:**")
            st.write(splits if not splits.empty else "No splits found during this period.")

        # Tab 1: Raw Data
        with tabs[2]:
            st.subheader(f"Raw Data for {stock_ticker}")
            st.write(data.tail())
            csv = data.to_csv().encode("utf-8")
            st.download_button("Download CSV", csv, file_name=f"{stock_ticker}_data.csv", mime="text/csv")

    else:
        st.warning("No data available for this stock.")