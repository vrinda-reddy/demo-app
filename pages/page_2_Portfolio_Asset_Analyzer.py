import yfinance as yf
import streamlit as st
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from datetime import timedelta
import requests
import json
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ssl


st.set_page_config(page_title="Portfolio Asset Analyzer", page_icon="📊", layout="wide")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_risk_free_rate():
    """Fetch 10-year US Treasury yield - cached for 1 hour"""
    try:
        hist = safe_yf_history("^TNX", period="5d")
        if not hist.empty:
            latest_close = hist['Close'].dropna().iloc[-1]
            return latest_close, f"{latest_close:.2f}%"
        else:
            return None, "N/A"
    except Exception as e:
        st.error(f"Error fetching risk-free rate: {e}")
        return None, "N/A"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_benchmark_data(ticker_to_download, start_date, end_date):
    """Fetch and process benchmark data with rate limiting protection"""
    from datetime import timedelta
    end_date_adjusted = end_date + timedelta(days=1)
    
    try:
        data_all = safe_yf_download(
            ticker_to_download,
            start=start_date,
            end=end_date_adjusted,
            progress=False,
            auto_adjust=False,
            group_by='ticker'
        )

        # Handle MultiIndex columns
        if (isinstance(data_all, pd.DataFrame) 
            and isinstance(data_all.columns, pd.MultiIndex) 
            and ticker_to_download in data_all.columns.get_level_values(0)):
            data_all = data_all[ticker_to_download]

        # Convert Series to DataFrame if needed
        if isinstance(data_all, pd.Series):
            data_all = data_all.to_frame()

        # Persist raw download
        st.session_state.data_all = data_all

        if not data_all.empty:
            if 'Adj Close' in data_all.columns:
                price_series = data_all['Adj Close']
            elif 'Close' in data_all.columns:
                price_series = data_all['Close']
            else:
                return None, None, "No price data found"
            
            # Native-currency benchmark series
            st.session_state.benchmark_prices = price_series

            # Get currency info using safe wrapper
            benchmark_info = safe_yf_ticker_info(ticker_to_download)
            benchmark_currency = benchmark_info.get("currency", "USD")
            
            return price_series, benchmark_currency, None
        else:
            return None, None, "No data returned"
            
    except Exception as e:
        return None, None, str(e)

@st.cache_data(ttl=3600)  # Cache for 1 hour  
def convert_to_usd(price_series, currency, start_date, end_date):
    """Convert price series to USD - cached for 1 hour"""
    if currency in ["USD", "N/A", None]:
        return price_series
        
    from datetime import timedelta
    end_date_adjusted = end_date + timedelta(days=1)
    fx_ticker = f"{currency}USD=X"
    
    try:
        fx_data = safe_yf_download(
            fx_ticker,
            start=start_date,
            end=end_date_adjusted,
            progress=False,
            auto_adjust=True
        )

        if isinstance(fx_data, pd.Series):
            fx_data = fx_data.to_frame()


        # Save FX series for downloads (keyed by ticker)
        st.session_state.fx_data = st.session_state.get("fx_data", {})
        st.session_state.fx_data[fx_ticker] = fx_data.copy()


        if not fx_data.empty and 'Close' in fx_data.columns:
            fx_rates_raw = fx_data['Close']

            if isinstance(fx_rates_raw, pd.DataFrame):
                fx_rates = fx_rates_raw.iloc[:, 0]
            elif isinstance(fx_rates_raw, pd.Series):
                fx_rates = fx_rates_raw
            else:
                return price_series

            fx_rates.index = pd.to_datetime(fx_rates.index)
            fx_rates = pd.to_numeric(fx_rates, errors='coerce')
            fx_rates_aligned = fx_rates.reindex(price_series.index).ffill()

            price_series = pd.to_numeric(price_series, errors='coerce')
            price_series_aligned = price_series.reindex(fx_rates_aligned.index)

            converted_prices = price_series_aligned * fx_rates_aligned
            return converted_prices
        else:
            return price_series
            
    except Exception as e:
        st.warning(f"Currency conversion failed: {e}")
        return price_series

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def calculate_market_return(price_series):
    """Calculate annualized return - cached for 30 minutes"""
    if price_series is None or price_series.empty or len(price_series) < 2:
        return np.nan

    # Clean data
    price_series = price_series[price_series > 0].dropna()
    if len(price_series) < 2:
        return np.nan

    prices = price_series.to_frame(name='price')
    start_date = price_series.index[0]
    end_date = price_series.index[-1]

    yearly_returns = []
    yearly_weights = []
    current_end = end_date

    # Process full years (weight = 1)
    while (current_end - pd.DateOffset(years=1)) >= start_date:
        current_start = current_end - pd.DateOffset(years=1)
        mask = (prices.index >= current_start) & (prices.index <= current_end)
        yearly_prices = prices.loc[mask]

        if len(yearly_prices) >= 2:
            start_price = yearly_prices.iloc[0]['price']
            end_price = yearly_prices.iloc[-1]['price']
            yearly_return = (end_price - start_price) / start_price
            yearly_returns.append(yearly_return)
            yearly_weights.append(1.0)

        current_end = current_start

    # Process partial year
    if current_end > start_date:
        mask = (prices.index >= start_date) & (prices.index <= current_end)
        partial_prices = prices.loc[mask]

        if len(partial_prices) >= 2:
            start_price = partial_prices.iloc[0]['price']
            end_price = partial_prices.iloc[-1]['price']
            partial_return = (end_price - start_price) / start_price
            partial_days = (current_end - start_date).days
            partial_weight = partial_days / 252
            
            yearly_returns.append(partial_return)
            yearly_weights.append(partial_weight)

    if not yearly_returns:
        return np.nan

    # Calculate weighted geometric mean
    weighted_product = 1.0
    total_weight = 0.0
    
    for ret, weight in zip(yearly_returns, yearly_weights):
        weighted_product *= (1 + ret) ** weight
        total_weight += weight

    if total_weight <= 0:
        return np.nan

    cagr = weighted_product ** (1 / total_weight) - 1
    return cagr

def clean_and_align_prices(prices_df):
    """Professional cleaning for both tables: 0→NaN, then limited forward-fill"""
    # Step 1: Convert 0s to NaN (market closures/holidays)
    clean_prices = prices_df.replace(0, np.nan)
    
    # Step 2: Forward-fill with 5-day limit (avoid overfilling suspensions)
    aligned_prices = clean_prices.ffill(limit=5)
    
    return aligned_prices

def get_earliest_available_data(ticker):
    try:
        hist = safe_yf_history(ticker, period="max")
        if not hist.empty:
            earliest_date = hist.index[0]
            return earliest_date
        else:
            return None
    except Exception as e:
        st.warning(f"Could not retrieve data for ticker '{ticker}': {e}")
        return None

def calculate_annualized_volatility(daily_returns):
    """Calculate annualized volatility from daily returns"""
    if len(daily_returns) < 2:
        return np.nan
    return np.std(daily_returns, ddof=1) * np.sqrt(252)

# def calculate_beta(stock_returns, benchmark_returns):
#     """Calculate beta for a stock relative to benchmark"""
#     if len(stock_returns) < 2 or len(benchmark_returns) < 2:
#         return np.nan

@st.cache_data(ttl=1800)
def is_valid_ticker_symbol(ticker):
    """Quick check if ticker symbol exists (lightweight validation)"""
    try:
        info = safe_yf_ticker_info(ticker)
        return 'symbol' in info or 'shortName' in info or 'longName' in info
    except:
        return False

def initialize_session_state():
    """Initialize all session state variables"""
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []
    if "portfolio_info" not in st.session_state:
        st.session_state.portfolio_info = {}
    if "last_add_error" not in st.session_state:
        st.session_state.last_add_error = ""
    if "data_processed" not in st.session_state:
        st.session_state.data_processed = False
    if "start_date" not in st.session_state:
        st.session_state.start_date = datetime.date.today() - timedelta(days=365) # 1 year ago
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.date.today()
    # NEW: Asset group variables
    if "enable_asset_groups" not in st.session_state:
        st.session_state.enable_asset_groups = False
    if "asset_groups" not in st.session_state:
        st.session_state.asset_groups = {}

def get_ticker_direct_api(company_name, max_results=8):
    """Direct Yahoo Finance API search - global search"""
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    params = {
        "q": company_name,
        "quotes_count": max_results
    }
    
    try:
        response = requests.get(
            url=url, 
            params=params, 
            headers={'User-Agent': user_agent},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        if 'quotes' in data and data['quotes']:
            for quote in data['quotes']:
                results.append({
                    'symbol': quote['symbol'],
                    'shortname': quote.get('shortName', ''),
                    'longname': quote.get('longName', ''),
                    'exchange': quote.get('exchDisp', ''),
                    'score': quote.get('score', 0),
                    'source': 'direct_api'
                })
        return results
        
    except Exception as e:
        print(f"Direct API search failed for '{company_name}': {e}")
        return []

def search_company_yfinance(company_name, max_results=8):
    """Modern yfinance.Search implementation"""
    try:
        search = yf.Search(company_name, max_results=max_results)
        quotes = search.quotes
        
        results = []
        for quote in quotes:
            results.append({
                'symbol': quote['symbol'],
                'shortname': quote.get('shortname', ''),
                'longname': quote.get('longname', ''),
                'exchange': quote.get('exchange', ''),
                'quoteType': quote.get('quoteType', ''),
                'source': 'yfinance_search'
            })
        return results
    except Exception as e:
        print(f"yfinance.Search failed for '{company_name}': {e}")
        return []

def safe_ticker_search(search_input):
    """Production-ready search with fallback methods - global search"""
    results = []
    
    # If input looks like a ticker (short, uppercase letters), validate it directly first
    if len(search_input) <= 6 and search_input.replace('.', '').replace('-', '').isalpha():
        # Try as direct ticker first
        if is_valid_ticker_symbol(search_input.upper()):
            try:
                info = safe_yf_ticker_info(search_input.upper())
                company_name = info.get('longName', info.get('shortName', search_input.upper()))
                
                results.append({
                    'symbol': search_input.upper(),
                    'longname': company_name,
                    'shortname': company_name,
                    'exchange': info.get('exchange', ''),
                    'source': 'direct_ticker'
                })
                return results  # If direct ticker works, return immediately
            except:
                pass  # Continue to search methods if direct ticker fails
    
    # Method 1: Try yfinance.Search first
    try:
        yf_results = search_company_yfinance(search_input)
        if yf_results:
            results.extend(yf_results)
    except Exception as e:
        print(f"yfinance.Search failed: {e}")
    
    # Method 2: Fallback to direct API if we don't have enough results
    if len(results) < 3:
        try:
            api_results = get_ticker_direct_api(search_input)
            
            # Avoid duplicates
            existing_symbols = {r['symbol'] for r in results}
            for result in api_results:
                if result['symbol'] not in existing_symbols:
                    results.append(result)
                    
        except Exception as e:
            print(f"Direct API also failed: {e}")
    
    # Remove duplicates and validate top results
    validated_results = []
    seen_symbols = set()
    
    for result in results[:10]:  # Limit to top 10 for performance
        symbol = result['symbol']
        if symbol not in seen_symbols:
            seen_symbols.add(symbol)
            # Quick validation for top results only
            if len(validated_results) < 5 and is_valid_ticker_symbol(symbol):
                validated_results.append(result)
            elif len(validated_results) >= 5:
                # Don't validate beyond top 5 for performance
                validated_results.append(result)
    
    return validated_results

def format_search_result(result):
    """Format search result for display"""
    symbol = result['symbol']
    longname = result.get('longname', '')
    shortname = result.get('shortname', '')
    exchange = result.get('exchange', '')
    
    # Choose best name
    display_name = longname if longname else shortname
    if not display_name:
        display_name = symbol
    
    # Format display string
    display_text = f"{display_name} ({symbol})"
    if exchange:
        display_text += f" - {exchange}"
    
    return display_text



def safe_yf_download(*args, **kwargs):
    """Simplified wrapper using yfinance 0.2.59's built-in protection"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(0.5)
            
            data = yf.download(*args, **kwargs)
            
            if data is not None and not data.empty:
                return data
            else:
                raise Exception("Empty data returned")
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"Failed to download data: {str(e)}")
                return pd.DataFrame()
    
    return pd.DataFrame()

def safe_yf_ticker_info(ticker_symbol):
    """Simplified ticker info wrapper"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(0.5)
            
            ticker = yf.Ticker(ticker_symbol)  # Removed session parameter
            info = ticker.info
            
            if info and isinstance(info, dict) and len(info) > 5:
                return info
            else:
                raise Exception("Invalid info returned")
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"Could not fetch info for {ticker_symbol}: {str(e)}")
                return {}
    
    return {}


def safe_yf_history(ticker_symbol, **kwargs):
    """Simplified history wrapper"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(0.5)
            
            ticker = yf.Ticker(ticker_symbol)  # Removed session parameter
            hist = ticker.history(**kwargs)
            
            if hist is not None and not hist.empty:
                return hist
            else:
                raise Exception("Empty history returned")
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"Could not fetch history for {ticker_symbol}: {str(e)}")
                return pd.DataFrame()
    
    return pd.DataFrame()

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

#Set allowed date range
earliest_allowed = datetime.date(1970, 1, 1)
today = datetime.date.today()

st.title('Portfolio Asset Selector & Analysis')
initialize_session_state()

# Emergency controls for page 2
with st.sidebar.expander("🚨 Data Loading Controls"):
    if st.button("🔄 Clear Cache & Reload", key="page2_reload"):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("⏸️ Use Offline Mode", key="page2_offline"):
        st.session_state.offline_mode = True
        st.rerun()

# Check offline mode
if st.session_state.get('offline_mode', False):
    st.warning("🔌 Offline Mode Active - Some data fetching will be limited")

st.info("📌 Tip: After adding assets, proceed to **Page 3: Portfolio Optimizer** to generate an optimized allocation.")

with st.expander("ℹ️ Notes on Portfolio Analysis and Optimization", expanded=False):
    st.markdown("""
    - This platform uses data from **Yahoo Finance**.
    - The **optimization on Page 3** is based on **daily price returns** of assets added on this page, over the **selected time period**.
    - You may select **tickers across different currencies and exchanges**. All prices are automatically converted to **USD using historical FX rates** to allow for fair comparison.
    - The **beta value** shown on this page is computed using **5-year daily returns** (or the maximum available history if shorter).
    - The results from the optimization **do not predict future returns** beyond the selected time period.
    - On the next page, you can apply the **Black-Litterman model** to express your own market views and generate a customized asset allocation.
    """)
    st.info("👉 Once you're done adding and analyzing assets, proceed to **Page 3: Portfolio Optimizer** to build your optimal portfolio.")


# Start date row
col1, col2 = st.columns([1, 2])
with col1:
    st.write("Start date")
with col2:
    start_date = st.date_input(
        "Select start date",
        value=st.session_state.start_date,
        min_value=earliest_allowed,
        max_value=today,
        label_visibility="collapsed",
        key="start_date_input"
    )
    st.session_state.start_date = start_date  # Save back to session_state
    st.write(f"**Selected (dd/mm/yyyy):** {start_date.strftime('%d/%m/%Y')}")

# End date row
col3, col4 = st.columns([1, 2])
with col3:
    st.write("End date")
with col4:
    end_date = st.date_input(
        "Select end date",
        value=st.session_state.end_date,
        min_value=earliest_allowed,
        max_value=today,
        label_visibility="collapsed",
        key="end_date_input"
    )
    st.session_state.end_date = end_date  # Save back to session_state
    st.write(f"**Selected (dd/mm/yyyy):** {end_date.strftime('%d/%m/%Y')}")
    

# Benchmark selection
benchmark_options = [
    'S&P 500 (^GSPC)', 'MSCI World (XWD.TO)', 
    'MSCI ACWI (ACWI)', 'FTSE All-World (VWRL.AS)', 'Specify ticker'
]

benchmark_dict = {
    'S&P 500 (^GSPC)': '^GSPC',
    'MSCI World (XWD.TO)': 'XWD.TO',
    'MSCI ACWI (ACWI)': 'ACWI',
    'FTSE All-World (VWRL.AS)': 'VWRL.AS'
}

col4, col5 = st.columns([1, 2])
with col4:
    st.write("Benchmark")
with col5:
    selected_benchmark = st.selectbox(
        "Select benchmark",
        benchmark_options,
        label_visibility="collapsed",
        key="benchmark_select"
    )
    
    if selected_benchmark == 'Specify ticker':
        user_ticker = st.text_input("Enter benchmark ticker (e.g., ^N225, ^GSPC, etc.)", key="custom_benchmark").strip().upper()
        ticker = user_ticker if user_ticker else None
    else:
        ticker = benchmark_dict.get(selected_benchmark)

    # Store the benchmark ticker in session state
    if ticker:
        st.session_state.benchmark_ticker = ticker    
    

price_series = None
benchmark_currency = "USD"
error_message = None
market_return_str = "N/A"

if ticker:
    if start_date > end_date:
        error_message = "Error: Start date cannot be after end date."
    else:
        # Defensive: ensure ticker is str or list of str
        if isinstance(ticker, (list, tuple)) and len(ticker) == 1:
            ticker_to_download = ticker[0]
        else:
            ticker_to_download = ticker
            
        # Use cached function for benchmark data
        price_series, benchmark_currency, fetch_error = get_benchmark_data(ticker_to_download, start_date, end_date)
        
        if fetch_error:
            error_message = f"Error fetching benchmark data: {fetch_error}"
        elif price_series is not None:
            # Convert to USD using cached function
            price_series_usd = convert_to_usd(price_series, benchmark_currency, start_date, end_date)
            
            # Calculate market return using cached function
            annual_return = calculate_market_return(price_series_usd)
            
            if isinstance(annual_return, (pd.Series, pd.DataFrame)):
                error_message = "Unable to calculate annualized return - result is not a scalar."
            elif np.isnan(annual_return):
                error_message = "Unable to calculate annualized return - check the date range."
            else:
                market_return_str = f"{annual_return:.2%}"
                
            # Store in session state
            st.session_state.benchmark_prices_usd = price_series_usd

            # Also get 5-year benchmark data for beta calculations
            five_years_ago = today - timedelta(days=5*365)
            end_date_adjusted_5y = today + timedelta(days=1)

            try:
                benchmark_5y_prices, benchmark_5y_currency, benchmark_5y_error = get_benchmark_data(ticker_to_download, five_years_ago, end_date_adjusted_5y)
                if benchmark_5y_error:
                    st.warning(f"Could not fetch 5-year benchmark data: {benchmark_5y_error}")
                    st.session_state.benchmark_prices_5y_usd = pd.Series()
                elif benchmark_5y_prices is not None:
                    benchmark_5y_usd = convert_to_usd(benchmark_5y_prices, benchmark_5y_currency, five_years_ago, end_date_adjusted_5y)
                    st.session_state.benchmark_prices_5y_usd = benchmark_5y_usd
                else:
                    st.session_state.benchmark_prices_5y_usd = pd.Series()
            except Exception as e:
                st.warning(f"Error fetching 5-year benchmark data: {e}")
                st.session_state.benchmark_prices_5y_usd = pd.Series()
        else:
            error_message = "No benchmark data available for the selected period."
else:
    error_message = "Please select or enter a valid benchmark ticker."

# Display Market Return
col_mr_label, col_mr_value = st.columns([1, 2])
with col_mr_label:
    st.write("Market return")
with col_mr_value:
    st.write(market_return_str)

if error_message:
    st.error(error_message)


# Display the currency-converted benchmark daily prices for selected period
# if "benchmark_prices_usd" in st.session_state and not st.session_state.benchmark_prices_usd.empty:
#     st.write("### Daily adjusted close prices for benchmark index in USD")
#     st.dataframe(st.session_state.benchmark_prices_usd.fillna(0))
# else:
#     st.info("Benchmark prices in USD not available for the selected period.")


def average_annualized_return(price_series):
    if price_series.empty or len(price_series) < 2:
        return np.nan

    # Clean data
    price_series = price_series[price_series > 0].dropna()
    if len(price_series) < 2:
        return np.nan

    prices = price_series.to_frame(name='price')
    start_date = price_series.index[0]
    end_date = price_series.index[-1]

    yearly_returns = []
    yearly_weights = []  # Track weights for each period
    current_end = end_date

    # Process full years (weight = 1)
    while (current_end - pd.DateOffset(years=1)) >= start_date:
        current_start = current_end - pd.DateOffset(years=1)
        mask = (prices.index >= current_start) & (prices.index <= current_end)
        yearly_prices = prices.loc[mask]

        if len(yearly_prices) >= 2:
            start_price = yearly_prices.iloc[0]['price']
            end_price = yearly_prices.iloc[-1]['price']
            yearly_return = (end_price - start_price) / start_price
            yearly_returns.append(yearly_return)
            yearly_weights.append(1.0)  # Full year weight = 1

        current_end = current_start

    # Process partial year (weight = fraction of year)
    if current_end > start_date:
        mask = (prices.index >= start_date) & (prices.index <= current_end)
        partial_prices = prices.loc[mask]

        if len(partial_prices) >= 2:
            start_price = partial_prices.iloc[0]['price']
            end_price = partial_prices.iloc[-1]['price']
            partial_return = (end_price - start_price) / start_price
            partial_days = (current_end - start_date).days
            partial_weight = partial_days / 252  # Fraction of year
            
            yearly_returns.append(partial_return)
            yearly_weights.append(partial_weight)

    if not yearly_returns:
        return np.nan

    # Calculate weighted geometric mean
    weighted_product = 1.0
    total_weight = 0.0
    
    for ret, weight in zip(yearly_returns, yearly_weights):
        weighted_product *= (1 + ret) ** weight
        total_weight += weight

    if total_weight <= 0:
        return np.nan

    cagr = weighted_product ** (1 / total_weight) - 1
    return cagr



# Fetch 10-year US Treasury yield using yfinance (^TNX returns yield * 10)
# Risk-free rate section with custom option
col_rf_label, col_rf_value = st.columns([1, 2])
with col_rf_label:
    st.write("Risk-free rate<br>(US 10-year Treasury yield)", unsafe_allow_html=True)
with col_rf_value:
    # Always show auto-fetched rate first
    auto_risk_free_rate, auto_risk_free_rate_str = get_risk_free_rate()
    st.write(auto_risk_free_rate_str)
    
    # Option to override below
    use_custom_rate = st.checkbox("Use custom risk-free rate", key="custom_rf_checkbox")
    
    if use_custom_rate:
        # Custom input
        custom_rate = st.number_input(
            "Enter risk-free rate (%)", 
            min_value=0.0, 
            max_value=20.0, 
            value=4.0, 
            step=0.01,
            key="custom_rf_input"
        )
        risk_free_rate = custom_rate
        risk_free_rate_str = f"{custom_rate:.2f}%"
        st.write(f"**Using custom rate:** {risk_free_rate_str}")
    else:
        # Use auto-fetched
        risk_free_rate = auto_risk_free_rate
        risk_free_rate_str = auto_risk_free_rate_str

#Select Tickers
    
st.subheader("Add stock tickers to your portfolio")

# Form Submission
with st.form("ticker_form", clear_on_submit=True):
    st.markdown("**Search Company Name or Ticker**")
    search_input = st.text_input(
        "",
        placeholder="e.g., Apple, Tesla, AAPL, TSLA",
        key="search_input",
        label_visibility="collapsed"
    )
    st.caption("Enter company name or ticker symbol")
    
    # Search button
    search_submitted = st.form_submit_button("Search Companies", type="secondary")
    
    # Add to portfolio section (only show if we have search results)
    if search_submitted and search_input:
        # Store search results in session state
        with st.spinner("Searching for companies..."):
            search_results = safe_ticker_search(search_input)
            st.session_state.search_results = search_results
            st.session_state.last_search_query = search_input

# Display search results outside the form
if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
    st.write("### Search Results")
    st.write(f"Found {len(st.session_state.search_results)} result(s) for '{st.session_state.last_search_query}':")
    
    # Display results as buttons
    for i, result in enumerate(st.session_state.search_results):
        display_text = format_search_result(result)
        ticker_symbol = result['symbol']
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.write(f"**{display_text}**")
            if result.get('source'):
                st.caption(f"Source: {result['source']}")
        
        with col2:
            if st.button("Add", key=f"add_{ticker_symbol}_{i}"):
                # Add to portfolio logic (reuse existing validation)
                st.session_state.last_add_error = ""
                existing = [s["ticker"] for s in st.session_state.portfolio]
                
                if ticker_symbol not in existing:
                    if is_valid_ticker_symbol(ticker_symbol):
                        try:
                            info = safe_yf_ticker_info(ticker_symbol)
                            company_name = info.get('longName', result.get('longname', ticker_symbol))
                            currency = info.get('currency', 'N/A')
                            
                            st.session_state.portfolio.append({
                                "ticker": ticker_symbol,
                                "min": 0.0,
                                "max": 100.0,
                                "group": "No Group"
                            })

                            st.session_state.portfolio_info[ticker_symbol] = {
                                "company_name": company_name,
                                "currency": currency
                            }
                            
                            st.success(f"Added {ticker_symbol} to portfolio!")
                            st.session_state.data_processed = False
                            
                            # Clear search results after successful add
                            st.session_state.search_results = []
                            st.rerun()
                            
                        except Exception as e:
                            st.session_state.last_add_error = f"Error adding {ticker_symbol}: {e}"
                    else:
                        st.session_state.last_add_error = f"'{ticker_symbol}' is not a valid ticker."
                else:
                    st.session_state.last_add_error = f"'{ticker_symbol}' already in portfolio."
    
    # Clear results button
    if st.button("Clear Results"):
        st.session_state.search_results = []
        st.rerun()

# Manual ticker entry option
with st.expander("Add ticker directly (if you know the symbol)"):
    manual_ticker = st.text_input("Enter ticker symbol directly:", placeholder="e.g., AAPL", key="manual_ticker_input")
    if st.button("Add Direct Ticker") and manual_ticker:
        # Use existing validation logic
        st.session_state.last_add_error = ""
        existing = [s["ticker"] for s in st.session_state.portfolio]
        ticker_to_add = manual_ticker.strip().upper()
        
        if ticker_to_add not in existing:
            if is_valid_ticker_symbol(ticker_to_add):
                try:
                    info = safe_yf_ticker_info(ticker_to_add)
                    company_name = info.get('longName', ticker_to_add)
                    currency = info.get('currency', 'N/A')
                    
                    st.session_state.portfolio.append({
                        "ticker": ticker_to_add,
                        "min": 0.0,
                        "max": 100.0,
                        "group": "No Group"
                    })

                    st.session_state.portfolio_info[ticker_to_add] = {
                        "company_name": company_name,
                        "currency": currency
                    }
                    
                    st.success(f"Added {ticker_to_add} to portfolio!")
                    st.session_state.data_processed = False
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.last_add_error = f"Error adding {ticker_to_add}: {e}"
            else:
                st.session_state.last_add_error = f"'{ticker_to_add}' is not a valid ticker."
        else:
            st.session_state.last_add_error = f"'{ticker_to_add}' already in portfolio."


# Show error message (if any)
if hasattr(st.session_state, 'last_add_error') and st.session_state.last_add_error:
    st.error(st.session_state.last_add_error)

# Show error message (if any) right after the form:
if st.session_state.last_add_error:
    st.error(st.session_state.last_add_error)

# Get current portfolio tickers
tickers = [s["ticker"] for s in st.session_state.portfolio]

# Show small error message if fewer than 2 tickers
if len(tickers) < 2 and len(tickers) > 0:  # show only if at least 1 ticker exists but less than 2
    st.warning("Please enter at least 2 tickers.")


# Always show portfolio constraints and summary regardless of ticker count
if st.session_state.portfolio:
    # Asset Groups Toggle
    st.session_state.enable_asset_groups = st.checkbox(
        "Enable Asset Group Constraints", 
        value=st.session_state.enable_asset_groups,
        key="asset_groups_toggle"
    )
    
    st.write("### Portfolio constraints")

    # Add custom CSS to make input boxes shorter
    st.markdown("""
    <style>
    div[data-testid="stNumberInput"] > div > div > input {
        height: 35px !important;
        min-height: 35px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.enable_asset_groups:
        edit_cols = st.columns([2, 2, 2, 2, 1])
        headers = ["Ticker", "Min Weight (%)", "Max Weight (%)", "Group", ""]
    else:
        edit_cols = st.columns([2, 2, 2, 1])
        headers = ["Ticker", "Min Weight (%)", "Max Weight (%)", ""]

    for col, header in zip(edit_cols, headers):
        col.write(f"**{header}**")
        
    # Each row without input box labels
    for i, stock in enumerate(st.session_state.portfolio):
        if st.session_state.enable_asset_groups:
            c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
        else:
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            
        with c1:
            st.write(stock['ticker'])
        with c2:
            min_val = st.number_input(
                "", 0.0, 100.0, stock["min"], 0.1, key=f"min_{stock['ticker']}", label_visibility="collapsed"
            )
        with c3:
            max_val = st.number_input(
                "", min_val, 100.0, stock["max"], 0.1, key=f"max_{stock['ticker']}", label_visibility="collapsed"
            )
        
        if st.session_state.enable_asset_groups:
            with c4:
                # Group dropdown
                group_options = ["No Group", "Group A", "Group B", "Group C", "Group D", "Group E", "Group F", "Group G", "Group H"]
                current_group = stock.get("group", "No Group")
                
                selected_group = st.selectbox(
                    "",
                    group_options,
                    index=group_options.index(current_group) if current_group in group_options else 0,
                    key=f"group_{stock['ticker']}",
                    label_visibility="collapsed"
                )
                stock["group"] = selected_group
            
            with c5:
                # YOUR EXISTING REMOVAL CODE HERE:
                if st.button("❌", key=f"remove_{stock['ticker']}"):
                    ticker_to_remove = stock['ticker']
                    
                    # Remove from portfolio
                    st.session_state.portfolio.pop(i)
                    st.session_state.portfolio_info.pop(ticker_to_remove, None)
                    
                    # Remove from returns dataframe if it exists
                    if hasattr(st.session_state, 'five_year_returns_df') and not st.session_state.five_year_returns_df.empty and ticker_to_remove in st.session_state.five_year_returns_df.columns:
                        st.session_state.five_year_returns_df.drop(ticker_to_remove, axis=1, inplace=True)
                    
                    # Remove from the prices dataframe
                    if hasattr(st.session_state, 'prices_5y_usd') and ticker_to_remove in st.session_state.prices_5y_usd.columns:
                        st.session_state.prices_5y_usd.drop(ticker_to_remove, axis=1, inplace=True)
                    
                    # Mark data as needing reprocessing
                    st.session_state.data_processed = False
                    
                    # Refresh the page after removal
                    st.rerun()
                    break
        else:
            with c4:
                # YOUR EXISTING REMOVAL CODE (when groups disabled):
                if st.button("❌", key=f"remove_{stock['ticker']}"):
                    ticker_to_remove = stock['ticker']
                    
                    # Remove from portfolio
                    st.session_state.portfolio.pop(i)
                    st.session_state.portfolio_info.pop(ticker_to_remove, None)
                    
                    # Remove from returns dataframe if it exists
                    if hasattr(st.session_state, 'five_year_returns_df') and not st.session_state.five_year_returns_df.empty and ticker_to_remove in st.session_state.five_year_returns_df.columns:
                        st.session_state.five_year_returns_df.drop(ticker_to_remove, axis=1, inplace=True)
                    
                    # Remove from the prices dataframe
                    if hasattr(st.session_state, 'prices_5y_usd') and ticker_to_remove in st.session_state.prices_5y_usd.columns:
                        st.session_state.prices_5y_usd.drop(ticker_to_remove, axis=1, inplace=True)
                    
                    # Mark data as needing reprocessing
                    st.session_state.data_processed = False
                    
                    # Refresh the page after removal
                    st.rerun()
                    break

        stock["min"] = min_val
        stock["max"] = max_val

    # Asset Groups Constraints Table (only if enabled and groups are being used)
    if st.session_state.enable_asset_groups:
        # Find which groups are actually being used
        used_groups = set()
        for stock in st.session_state.portfolio:
            group = stock.get("group", "No Group")
            if group != "No Group":
                used_groups.add(group)
        
        if used_groups:
            st.write("### Asset Group Constraints")

            # Add custom CSS to make input boxes shorter
            st.markdown("""
            <style>
            div[data-testid="stNumberInput"] > div > div > input {
                height: 35px !important;
                min-height: 35px !important;
            }
            div[data-testid="stTextInput"] > div > div > input {
                height: 35px !important;
                min-height: 35px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Headers for group table
            group_cols = st.columns([3, 2, 2])
            group_headers = ["Group Name", "Min Weight (%)", "Max Weight (%)"]
            for col, header in zip(group_cols, group_headers):
                col.write(f"**{header}**")
            
            # Row for each used group
            for group in sorted(used_groups):
                gc1, gc2, gc3 = st.columns([3, 2, 2])
                
                with gc1:
                    # Group name input (optional)
                    if group not in st.session_state.asset_groups:
                        st.session_state.asset_groups[group] = {"name": "", "min": 0.0, "max": 100.0}
                    
                    group_name = st.text_input(
                        f"",
                        value=st.session_state.asset_groups[group]["name"],
                        placeholder=f"{group} (optional name)",
                        key=f"name_{group}",
                        label_visibility="collapsed"
                    )
                    st.session_state.asset_groups[group]["name"] = group_name
                
                with gc2:
                    # Group min weight
                    group_min = st.number_input(
                        "",
                        0.0, 100.0,
                        value=st.session_state.asset_groups[group]["min"],
                        step=0.1,
                        key=f"group_min_{group}",
                        label_visibility="collapsed"
                    )
                    st.session_state.asset_groups[group]["min"] = group_min
                
                with gc3:
                    # Group max weight
                    group_max = st.number_input(
                        "",
                        group_min, 100.0,
                        value=st.session_state.asset_groups[group]["max"],
                        step=0.1,
                        key=f"group_max_{group}",
                        label_visibility="collapsed"
                    )
                    st.session_state.asset_groups[group]["max"] = group_max
        
        # Clean up unused groups from session state
        groups_to_remove = []
        for group in st.session_state.asset_groups:
            if group not in used_groups:
                groups_to_remove.append(group)
        for group in groups_to_remove:
            del st.session_state.asset_groups[group]

# Get current portfolio tickers
tickers = [s["ticker"] for s in st.session_state.portfolio]

# Show processing button
if tickers and len(tickers) >= 1:
    st.write("---")  # Add a separator line
    col_button, col_info = st.columns([1, 2])
    
    with col_button:
        process_data = st.button("🚀 Get Portfolio Summary and Risk Metrics!", type="primary")
    
    with col_info:
        if len(tickers) < 2:
            st.info("Add at least 2 tickers to calculate risk metrics.")
        else:
            st.info(f"Ready to process {len(tickers)} tickers")
    
    # STEP 7: Wrap ALL your heavy processing in this condition
    if process_data:
        st.session_state.data_processed = True
        
        # Add progress indicator
        with st.spinner("Processing portfolio data... This may take a moment."):
            
            # CREATE A CONTAINER for validation messages that we can clear later
            with st.status("🔍 Validating ticker date ranges...", expanded=True) as status:
    
                tickers_to_remove = []
                valid_tickers = []
                
                for stock in st.session_state.portfolio:
                    ticker = stock["ticker"]
                    
                    # Now do the thorough date validation
                    earliest_data = get_earliest_available_data(ticker)
                    
                    if earliest_data:
                        earliest_data_naive = earliest_data.replace(tzinfo=None)
                        start_date_naive = pd.to_datetime(st.session_state.start_date).replace(tzinfo=None)

                        if earliest_data_naive > start_date_naive:
                            st.warning(f"⚠️ Removing {ticker}: data starts {earliest_data.strftime('%Y-%m-%d')}, after your selected start date ({st.session_state.start_date})")
                            tickers_to_remove.append(ticker)
                        else:
                            valid_tickers.append(ticker)
                            st.info(f"✅ {ticker}: data available from {earliest_data.strftime('%Y-%m-%d')}")
                    else:
                        st.warning(f"⚠️ Removing {ticker}: could not retrieve data")
                        tickers_to_remove.append(ticker)
                
                # Remove invalid tickers from portfolio
                if tickers_to_remove:
                    st.session_state.portfolio = [s for s in st.session_state.portfolio 
                                                if s["ticker"] not in tickers_to_remove]
                    for ticker in tickers_to_remove:
                        st.session_state.portfolio_info.pop(ticker, None)
                
                # Update tickers list with only valid ones
                tickers = [s["ticker"] for s in st.session_state.portfolio]
                
                if not tickers:
                    st.error("❌ No valid tickers remaining to process!")
                    st.stop()  # Stop processing
                
                st.success(f"✅ Proceeding with {len(tickers)} valid ticker(s): {', '.join(tickers)}")
                
                # Update tickers list for all subsequent processing
                tickers = [s["ticker"] for s in st.session_state.portfolio]

                # Clean existing DataFrames to remove invalid tickers
                if hasattr(st.session_state, 'five_year_returns_df') and not st.session_state.five_year_returns_df.empty:
                    invalid_columns = [col for col in st.session_state.five_year_returns_df.columns if col not in tickers]
                    if invalid_columns:
                        st.session_state.five_year_returns_df.drop(invalid_columns, axis=1, inplace=True)

                if hasattr(st.session_state, 'prices_5y_usd') and not st.session_state.prices_5y_usd.empty:
                    invalid_columns = [col for col in st.session_state.prices_5y_usd.columns if col not in tickers]
                    if invalid_columns:
                        st.session_state.prices_5y_usd.drop(invalid_columns, axis=1, inplace=True)

                if hasattr(st.session_state, 'portfolio_prices_usd_daily') and not st.session_state.portfolio_prices_usd_daily.empty:
                    invalid_columns = [col for col in st.session_state.portfolio_prices_usd_daily.columns if col not in tickers]
                    if invalid_columns:
                        st.session_state.portfolio_prices_usd_daily.drop(invalid_columns, axis=1, inplace=True)                        

                # Validate group constraints (if enabled)
                if st.session_state.enable_asset_groups and st.session_state.asset_groups:
                    validation_errors = []
                    
                    for group_name, group_data in st.session_state.asset_groups.items():
                        # Find all tickers in this group
                        tickers_in_group = [s["ticker"] for s in st.session_state.portfolio if s.get("group") == group_name]
                        
                        if tickers_in_group:
                            # Calculate sum of individual ticker minimums and maximums
                            total_min = sum([s["min"] for s in st.session_state.portfolio if s.get("group") == group_name])
                            total_max = sum([s["max"] for s in st.session_state.portfolio if s.get("group") == group_name])
                            
                            group_min = group_data["min"]
                            group_max = group_data["max"]
                            
                            # Check for conflicts
                            if total_min > group_max:
                                validation_errors.append(f"❌ {group_name}: Sum of individual minimums ({total_min:.1f}%) exceeds group maximum ({group_max:.1f}%)")
                            
                            if group_min > total_max:
                                validation_errors.append(f"❌ {group_name}: Group minimum ({group_min:.1f}%) exceeds sum of individual maximums ({total_max:.1f}%)")
                    
                    # If there are validation errors, stop processing
                    if validation_errors:
                        st.error("**Group Constraint Conflicts Detected:**")
                        for error in validation_errors:
                            st.error(error)
                        st.info("💡 **How to fix:** Adjust either the individual ticker constraints or the group constraints to resolve conflicts.")
                        st.stop()  # Stop processing
                    else:
                        st.success("✅ Group constraints validation passed!")

                # Mark status as complete  
                status.update(label="✅ Validation complete!", state="complete", expanded=False)


        # ------------ Portfolio tickers and prices for selected period ------------
    
            
            if tickers:
                end_date_adjusted = end_date + timedelta(days=1)
            
                # Download adjusted close prices for all tickers
                all_data = safe_yf_download(
                    tickers,
                    start=start_date,
                    end=end_date_adjusted,
                    progress=False,
                    auto_adjust=False
                )
            
                portfolio_prices_df = pd.DataFrame()  # default empty
            
                # Extract Adj Close prices; handle single vs multiple tickers
                if len(tickers) == 1:
                    ticker_sym = tickers[0]
                    if isinstance(all_data, pd.Series):
                        portfolio_prices_df = all_data.to_frame(name=ticker_sym)
                    elif isinstance(all_data, pd.DataFrame):
                        if 'Adj Close' in all_data.columns:
                            portfolio_prices_df = all_data[['Adj Close']].rename(columns={'Adj Close': ticker_sym})
                        elif 'Close' in all_data.columns:
                            portfolio_prices_df = all_data[['Close']].rename(columns={'Close': ticker_sym})
                else:
                    if 'Adj Close' in all_data.columns.get_level_values(0):
                        portfolio_prices_df = all_data['Adj Close']
                    elif 'Close' in all_data.columns.get_level_values(0):
                        portfolio_prices_df = all_data['Close']
            
                # Ensure index is DatetimeIndex
                if not pd.api.types.is_datetime64_any_dtype(portfolio_prices_df.index):
                    portfolio_prices_df.index = pd.to_datetime(portfolio_prices_df.index)

                if not portfolio_prices_df.empty:

                    # ⬇️ ADD THESE TWO (local-currency prices for selected period)
                    st.session_state.prices_local = portfolio_prices_df
                    st.session_state.portfolio_prices_df = portfolio_prices_df

                    # Make a copy for USD prices modifications
                    portfolio_prices_usd_df = portfolio_prices_df.copy()

                    # Map tickers to their currencies
                    currency_map = {
                        s["ticker"]: st.session_state.portfolio_info.get(s["ticker"], {}).get('currency', 'N/A')
                        for s in st.session_state.portfolio
                    }

                    # List of tickers that need conversion (non-USD)
                    non_usd_tickers = [
                        ticker for ticker, currency in currency_map.items()
                        if currency != 'USD' and currency not in ['N/A', None]
                    ]

                    for ticker in non_usd_tickers:
                        currency = currency_map[ticker]
                        fx_ticker = f"{currency}USD=X"
                        
                        fx_data = safe_yf_download(
                            fx_ticker,
                            start=start_date,
                            end=end_date_adjusted,
                            progress=False,
                            auto_adjust=True
                        )

                        # Persist FX series for downloads
                        st.session_state.fx_data = st.session_state.get("fx_data", {})
                        st.session_state.fx_data[fx_ticker] = fx_data.copy()


                        if not fx_data.empty and 'Close' in fx_data.columns and ticker in portfolio_prices_usd_df.columns:
                            fx_rates_raw = fx_data['Close']

                            # Ensure fx_rates is Series
                            if isinstance(fx_rates_raw, pd.DataFrame):
                                fx_rates = fx_rates_raw.iloc[:, 0]
                            elif isinstance(fx_rates_raw, pd.Series):
                                fx_rates = fx_rates_raw
                            else:
                                st.warning(f"FX rates for {fx_ticker} are not Series or DataFrame. Skipping conversion for {ticker}.")
                                continue

                            fx_rates.index = pd.to_datetime(fx_rates.index)
                            fx_rates = pd.to_numeric(fx_rates, errors='coerce')

                            # Align FX rates to stock prices index, forward fill missing
                            fx_rates_aligned = fx_rates.reindex(portfolio_prices_usd_df.index).ffill()

                            price_series = portfolio_prices_usd_df[ticker].copy()
                            price_series.index = pd.to_datetime(price_series.index)
                            price_series = pd.to_numeric(price_series, errors='coerce')
                            price_series_aligned = price_series.reindex(fx_rates_aligned.index)

                            converted_prices = price_series_aligned * fx_rates_aligned

                            portfolio_prices_usd_df.loc[:, ticker] = converted_prices
                        else:
                            st.warning(f"FX data empty or missing 'Close' for {fx_ticker}, or '{ticker}' not in price DataFrame.")

                    # Store final daily USD prices DataFrame for selected period
                    st.session_state.portfolio_prices_usd_daily = portfolio_prices_usd_df

                    # ⬇️ ADD THIS (keep a named copy for downloads)
                    st.session_state.portfolio_prices_usd_df = portfolio_prices_usd_df

                    # Calculate and store selected period returns
                    if not portfolio_prices_usd_df.empty:
                        # Calculate returns for the selected period
                        selected_period_prices = clean_and_align_prices(portfolio_prices_usd_df)
                        selected_period_returns = selected_period_prices.pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)
                        
                        # Remove any rows where ALL returns are zero (market holidays)
                        all_zero_mask = (selected_period_returns == 0).all(axis=1)
                        if all_zero_mask.sum() > 0:
                            selected_period_returns = selected_period_returns[~all_zero_mask]
                        
                        # Store in session state for use by Portfolio Optimizer
                        st.session_state.portfolio_returns_selected_period = selected_period_returns
                    else:
                        st.session_state.portfolio_returns_selected_period = pd.DataFrame()
                        
                    # OPTIONAL: Compute Start and End prices summary in USD
                    summary = pd.DataFrame(index=tickers, columns=['Start Price USD', 'End Price USD', 'Currency'])

                    for ticker_sym in tickers:
                        if ticker_sym in portfolio_prices_usd_df.columns:
                            series = portfolio_prices_usd_df[ticker_sym].dropna()
                            if not series.empty:
                                start_ts = pd.Timestamp(start_date)
                                end_ts = pd.Timestamp(end_date)

                                start_idx = series.index.asof(start_ts)
                                end_idx = series.index.asof(end_ts)

                                summary.at[ticker_sym, 'Start Price USD'] = series.loc[start_idx] if pd.notna(start_idx) else np.nan
                                summary.at[ticker_sym, 'End Price USD'] = series.loc[end_idx] if pd.notna(end_idx) else np.nan
                            else:
                                summary.at[ticker_sym, 'Start Price USD'] = np.nan
                                summary.at[ticker_sym, 'End Price USD'] = np.nan
                        else:
                            summary.at[ticker_sym, 'Start Price USD'] = np.nan
                            summary.at[ticker_sym, 'End Price USD'] = np.nan

                        summary.at[ticker_sym, 'Currency'] = currency_map.get(ticker_sym, 'N/A')

                    st.session_state.portfolio_prices_usd_summary = summary

                else:
                    st.session_state.portfolio_prices_usd_daily = pd.DataFrame()
                    st.session_state.portfolio_prices_usd_summary = pd.DataFrame()

            else:
                st.session_state.portfolio_prices_usd_daily = pd.DataFrame()
                st.session_state.portfolio_prices_usd_summary = pd.DataFrame()

            # ----- SHOW TABLE FOR SELECTED PERIOD HERE -----
            # Daily Adjusted Close Prices (For period selected) in USD
            if "portfolio_prices_usd_daily" in st.session_state and not st.session_state.portfolio_prices_usd_daily.empty:
                cleaned_prices = clean_and_align_prices(st.session_state.portfolio_prices_usd_daily)
                
                # Fix: Ensure proper chronological ordering
                # Convert index to datetime and sort
                cleaned_prices.index = pd.to_datetime(cleaned_prices.index)
                cleaned_prices = cleaned_prices.sort_index()
                
                # Remove duplicate index entries (keep last occurrence)
                cleaned_prices = cleaned_prices[~cleaned_prices.index.duplicated(keep='last')]
                
                # st.write("### Daily Adjusted Close Prices (For period selected) in USD")
                # st.dataframe(cleaned_prices)
                
            # ------------ 5-YEAR DATA --------------

            # Calculate 5 years ago date
            five_years_ago = today - timedelta(days=5*365)
            end_date_adjusted_5y = today + timedelta(days=1)  # ensure inclusive

            prices_5y_df = pd.DataFrame()

            if tickers:
                # First try to download all tickers together
                try:
                    data_5y = safe_yf_download(
                        tickers,
                        start=five_years_ago,
                        end=end_date_adjusted_5y,
                        progress=False,
                        auto_adjust=False
                    )
                except Exception as e:
                    st.warning(f"Batch download failed: {str(e)}. Falling back to individual downloads.")
                    data_5y = None

                # If batch download failed or returned empty, try individual downloads
                if data_5y is None or data_5y.empty:
                    data_5y = {}
                    for ticker in tickers:
                        try:
                            ticker_data = safe_yf_download(
                                ticker,
                                start=five_years_ago,
                                end=end_date_adjusted_5y,
                                progress=False,
                                auto_adjust=False
                            )
                            if not ticker_data.empty:
                                data_5y[ticker] = ticker_data
                        except Exception as e:
                            st.warning(f"Failed to download {ticker}: {str(e)}")
                    
                    # Combine individual downloads into multi-index dataframe
                    if data_5y:
                        prices_5y_df = pd.concat(data_5y.values(), axis=1, keys=data_5y.keys())
                    else:
                        prices_5y_df = pd.DataFrame()
                else:
                    # Extract Adj Close prices from batch download
                    if len(tickers) == 1:
                        ticker_sym = tickers[0]
                        if isinstance(data_5y, pd.Series):
                            prices_5y_df = data_5y.to_frame(name=ticker_sym)
                        elif isinstance(data_5y, pd.DataFrame):
                            if 'Adj Close' in data_5y.columns:
                                prices_5y_df = data_5y[['Adj Close']].rename(columns={'Adj Close': ticker_sym})
                            elif 'Close' in data_5y.columns:
                                prices_5y_df = data_5y[['Close']].rename(columns={'Close': ticker_sym})
                    else:
                        if 'Adj Close' in data_5y.columns.get_level_values(0):
                            prices_5y_df = data_5y['Adj Close']
                        elif 'Close' in data_5y.columns.get_level_values(0):
                            prices_5y_df = data_5y['Close']

                # Ensure index is DatetimeIndex
                if not prices_5y_df.empty:
                    if not pd.api.types.is_datetime64_any_dtype(prices_5y_df.index):
                        prices_5y_df.index = pd.to_datetime(prices_5y_df.index)

                    # Map tickers to their currencies
                    currency_map = {
                        s["ticker"]: st.session_state.portfolio_info.get(s["ticker"], {}).get('currency', 'N/A')
                        for s in st.session_state.portfolio
                    }

                    # Fallback for tickers missing data - now handles partial data
                    for ticker in tickers:
                        # Check if ticker exists in the dataframe
                        if ticker not in prices_5y_df.columns:
                            st.warning(f"No data for {ticker} from yf.download(). Using yf.Ticker().history() fallback.")
                            try:
                                hist = safe_yf_history(ticker, period="max")
                                
                                # Filter for our date range (but keep whatever is available)
                                hist = hist[hist.index <= pd.to_datetime(end_date_adjusted_5y)]
                                if not hist.empty and 'Close' in hist.columns:
                                    if isinstance(prices_5y_df, pd.DataFrame):
                                        prices_5y_df[ticker] = hist['Close']
                                    else:
                                        prices_5y_df = hist[['Close']].rename(columns={'Close': ticker})
                                    st.success(f"Successfully retrieved fallback data for {ticker} (max available period)")
                                else:
                                    st.warning(f"No fallback data found for {ticker}")
                            except Exception as e:
                                st.error(f"Fallback failed for {ticker}: {str(e)}")
                        else:
                            # Check if we have at least some data points
                            if prices_5y_df[ticker].isnull().all():
                                st.warning(f"No valid price data for {ticker} in selected period. Trying to get max available history.")
                                try:
                                    hist = safe_yf_history(ticker, period="max")
                                    hist = hist[hist.index <= pd.to_datetime(end_date_adjusted_5y)]
                                    if not hist.empty and 'Close' in hist.columns:
                                        prices_5y_df[ticker] = hist['Close']
                                        st.success(f"Updated {ticker} with max available history")
                                except Exception as e:
                                    st.error(f"Failed to update {ticker}: {str(e)}")

                    # ⬇️ ADD THIS (persist local-currency 5y prices)
                    st.session_state.prices_5y_df = prices_5y_df

                    # Now proceed with USD conversion for whatever data we have
                    if not prices_5y_df.empty:
                        prices_5y_usd_df = prices_5y_df.copy()

                        non_usd_tickers = [
                            ticker for ticker, currency in currency_map.items()
                            if currency != 'USD' and currency not in ['N/A', None] and ticker in prices_5y_usd_df.columns
                        ]

                        for ticker in non_usd_tickers:
                            currency = currency_map[ticker]
                            fx_ticker = f"{currency}USD=X"
                            
                            try:
                                fx_data = safe_yf_download(
                                    fx_ticker,
                                    start=prices_5y_usd_df.index[0],  # Use actual available start date
                                    end=end_date_adjusted_5y,
                                    progress=False,
                                    auto_adjust=True
                                )

                                # Persist FX series for downloads (5y)
                                st.session_state.fx_data = st.session_state.get("fx_data", {})
                                st.session_state.fx_data[fx_ticker] = fx_data.copy()


                                if not fx_data.empty and 'Close' in fx_data.columns:
                                    fx_rates_raw = fx_data['Close']

                                    if isinstance(fx_rates_raw, pd.DataFrame):
                                        fx_rates = fx_rates_raw.iloc[:, 0]
                                    elif isinstance(fx_rates_raw, pd.Series):
                                        fx_rates = fx_rates_raw
                                    else:
                                        st.warning(f"FX rates for {fx_ticker} are not Series or DataFrame. Skipping conversion for {ticker}.")
                                        continue

                                    fx_rates.index = pd.to_datetime(fx_rates.index)
                                    fx_rates = pd.to_numeric(fx_rates, errors='coerce')

                                    # Align FX rates to stock prices index, forward fill missing
                                    fx_rates_aligned = fx_rates.reindex(prices_5y_usd_df.index).ffill()

                                    price_series = prices_5y_usd_df[ticker].copy()
                                    price_series.index = pd.to_datetime(price_series.index)
                                    price_series = pd.to_numeric(price_series, errors='coerce')
                                    price_series_aligned = price_series.reindex(fx_rates_aligned.index)

                                    converted_prices = price_series_aligned * fx_rates_aligned

                                    prices_5y_usd_df.loc[:, ticker] = converted_prices
                                else:
                                    st.warning(f"FX data empty or missing 'Close' for {fx_ticker}. Prices remain in original currency.")
                            except Exception as e:
                                st.warning(f"Failed to convert {ticker} to USD: {str(e)}. Prices remain in original currency.")

                        # Store final daily USD prices DataFrame for whatever period we have
                        st.session_state.prices_5y_usd = prices_5y_usd_df

                        # ⬇️ ADD THIS (keep a second session key with the _df name)
                        st.session_state.prices_5y_usd_df = prices_5y_usd_df

                        # Add information about data availability
                        availability_info = {}
                        for ticker in tickers:
                            if ticker in prices_5y_usd_df.columns:
                                ticker_data = prices_5y_usd_df[ticker].dropna()
                                if not ticker_data.empty:
                                    start_date_actual = ticker_data.index[0].strftime('%Y-%m-%d')
                                    end_date_actual = ticker_data.index[-1].strftime('%Y-%m-%d')
                                    days_available = len(ticker_data)
                                    availability_info[ticker] = {
                                        'Start Date': start_date_actual,
                                        'End Date': end_date_actual,
                                        'Days Available': days_available
                                    }
                        
                        st.session_state.data_availability = availability_info
                    else:
                        st.session_state.prices_5y_usd = pd.DataFrame()
                        st.session_state.data_availability = {}
                else:
                    st.session_state.prices_5y_usd = pd.DataFrame()
                    st.session_state.data_availability = {}
            else:
                st.session_state.prices_5y_usd = pd.DataFrame()
                st.session_state.data_availability = {}

            # Display data availability information
            if hasattr(st.session_state, 'data_availability') and st.session_state.data_availability:
                #st.write("### Data Availability Information")
                
                # Prepare a list of tickers to collect the earliest data
                data_availability_info = []

                for ticker in st.session_state.data_availability:
                    # Get the earliest available date for each ticker
                    earliest_data = get_earliest_available_data(ticker)
                    
                    # Add information to the data_availability_info list
                    data_availability_info.append({
                        'Start Date': earliest_data.strftime('%Y-%m-%d') if earliest_data else 'N/A',
                        'End Date': st.session_state.data_availability[ticker].get('End Date', 'N/A'),
                        'Days Available': st.session_state.data_availability[ticker].get('Days Available', 'N/A')
                    })
                
                # Create DataFrame from the information collected
                availability_df = pd.DataFrame(data_availability_info, index=st.session_state.data_availability.keys())
                
                # Display the dataframe
                #st.dataframe(availability_df)

            # 1. First check if we have 5-year price data
            if hasattr(st.session_state, 'prices_5y_usd') and not st.session_state.prices_5y_usd.empty:
                
                # 2. Calculate returns from the prices
                # Initialize returns dataframe if it doesn't exist
                if "five_year_returns_df" not in st.session_state:
                    st.session_state.five_year_returns_df = pd.DataFrame()

                # Process new tickers for returns calculation
                existing_returns_tickers = st.session_state.five_year_returns_df.columns.tolist()
                new_tickers = [t for t in st.session_state.prices_5y_usd.columns 
                              if t not in existing_returns_tickers]
                
                for ticker in new_tickers:
                    price_series = st.session_state.prices_5y_usd[ticker].copy()
                    price_series.replace(0, np.nan, inplace=True)
                    price_series.dropna(inplace=True)
                    returns_series = price_series.pct_change().dropna()
                    
                    if st.session_state.five_year_returns_df.empty:
                        st.session_state.five_year_returns_df = pd.DataFrame({ticker: returns_series})
                    else:
                        aligned_returns = returns_series.reindex(st.session_state.five_year_returns_df.index)
                        st.session_state.five_year_returns_df[ticker] = aligned_returns

                st.session_state.five_year_returns_df.dropna(how='all', inplace=True)
                
                # 3. Display returns
                if not st.session_state.five_year_returns_df.empty:
                    #st.write("### 5-Year Daily Returns")
                    
                    # Fill NaNs with 0.0 directly in the existing dataframe
                    st.session_state.five_year_returns_df.fillna(0.0, inplace=True)
                    
                    # Format to show percentages and handle zero values
                    def format_return(val):
                        if val == 0.0:
                            return "0.00%"  # Display zero returns as "0.00%"
                        return f"{val:.2%}"  # Format non-zero returns as percentages
                    
                    # Apply formatting to all columns
                    formatted_returns = st.session_state.five_year_returns_df.applymap(format_return)
                    
                    #st.dataframe(formatted_returns)

                # 4. Display prices
                # Display Daily Adjusted Close Prices (Past 5 Years) in USD
                cleaned_5y_prices = clean_and_align_prices(st.session_state.prices_5y_usd)
                #st.write("### Daily Adjusted Close Prices (Past 5 Years) in USD")
                #st.dataframe(cleaned_5y_prices)

            # ====== COMPLETE BETA CALCULATION CODE ======
            if hasattr(st.session_state, 'five_year_returns_df') and not st.session_state.five_year_returns_df.empty and len(tickers) >= 2:
                
                # 1. Get benchmark returns for the full 5-year period
                benchmark_returns_full = pd.Series(dtype=float)
                if "benchmark_prices_5y_usd" in st.session_state and not st.session_state.benchmark_prices_5y_usd.empty:
                    benchmark_prices = st.session_state.benchmark_prices_5y_usd.copy()
                    benchmark_prices.replace(0, np.nan, inplace=True)
                    benchmark_returns_full = benchmark_prices.pct_change().dropna()
                    # Store benchmark returns for download
                    st.session_state.benchmark_returns_5y = benchmark_returns_full
                else:
                    st.warning("No 5-year benchmark data available for beta calculation")   

                # 2. Calculate benchmark volatility for the full period
                if not benchmark_returns_full.empty:
                    benchmark_volatility = np.std(benchmark_returns_full, ddof=1) * np.sqrt(252)
                else:
                    benchmark_volatility = np.nan
                    st.warning("Cannot calculate benchmark volatility")

                # 3. Prepare to store risk metrics
                risk_metrics = []

                # 4. Calculate metrics for each stock
                for ticker in st.session_state.five_year_returns_df.columns:
                    try:
                        # Get stock returns
                        stock_returns = st.session_state.five_year_returns_df[ticker].dropna()
                        
                        if len(stock_returns) < 2:
                            st.warning(f"Insufficient data for {ticker}")
                            continue
                        
                        # Calculate stock volatility (annualized)
                        stock_volatility = np.std(stock_returns, ddof=1) * np.sqrt(252)
                        
                        # Get the date range for this stock's available data
                        stock_start_date = stock_returns.index[0]
                        stock_end_date = stock_returns.index[-1]
                        
                        # Get corresponding benchmark returns for the same period
                        benchmark_returns_aligned = benchmark_returns_full.loc[stock_start_date:stock_end_date]
                        
                        # Align both series to have the same dates
                        common_dates = stock_returns.index.intersection(benchmark_returns_aligned.index)
                        stock_returns_aligned = stock_returns.loc[common_dates]
                        benchmark_returns_matched = benchmark_returns_aligned.loc[common_dates]
                        
                        if len(stock_returns_aligned) < 2 or len(benchmark_returns_matched) < 2:
                            st.warning(f"Insufficient aligned data for {ticker}")
                            continue
                        
                        # Calculate correlation
                        correlation = np.corrcoef(stock_returns_aligned, benchmark_returns_matched)[0, 1]
                        
                        # Calculate benchmark volatility for the matched period
                        benchmark_volatility_matched = np.std(benchmark_returns_matched, ddof=1) * np.sqrt(252)
                        
                        # Calculate beta
                        if benchmark_volatility_matched != 0 and not np.isnan(benchmark_volatility_matched):
                            beta = correlation * (stock_volatility / benchmark_volatility_matched)
                        else:
                            beta = np.nan
                        
                        # Calculate Sharpe Ratio directly from available data
                        ticker_sharpe_ratio = np.nan
                        if hasattr(st.session_state, 'portfolio_prices_usd_daily') and not st.session_state.portfolio_prices_usd_daily.empty:
                            if ticker in st.session_state.portfolio_prices_usd_daily.columns:
                                ticker_prices = st.session_state.portfolio_prices_usd_daily[ticker].dropna()
                                if not ticker_prices.empty and len(ticker_prices) >= 2:
                                    try:
                                        # Calculate actual return using the same function as portfolio summary
                                        actual_return = average_annualized_return(ticker_prices)
                                        if not np.isnan(actual_return) and stock_volatility > 0:
                                            # Get risk-free rate as decimal
                                            risk_free_decimal = float(risk_free_rate_str.replace('%', '')) / 100 if risk_free_rate_str != "N/A" else 0.04
                                            # Calculate Sharpe ratio: (Return - Risk-free rate) / Volatility
                                            ticker_sharpe_ratio = (actual_return - risk_free_decimal) / stock_volatility
                                    except:
                                        ticker_sharpe_ratio = np.nan

                        # Store metrics
                        risk_metrics.append({
                            'Ticker': ticker,
                            'Beta': beta,
                            'Volatility (Annualized)': stock_volatility,
                            'Correlation with Benchmark': correlation,
                            'Sharpe Ratio': ticker_sharpe_ratio
                        })                                                
                        
                    except Exception as e:
                        st.warning(f"Error calculating metrics for {ticker}: {str(e)}")
                        continue

                # 5. Store results (don't display yet)
                if risk_metrics:
                    # Create DataFrame
                    risk_metrics_df = pd.DataFrame(risk_metrics)
                    
                    # Store in session state for future use
                    st.session_state.risk_metrics_df = risk_metrics_df
                    st.session_state.benchmark_volatility = benchmark_volatility
                    st.session_state.selected_benchmark_name = selected_benchmark
                    
                else:
                    st.session_state.risk_metrics_df = pd.DataFrame()
                    
            else:
                # This else belongs to the outer if statement (when tickers < 2 or no data)
                st.session_state.risk_metrics_df = pd.DataFrame()

            # Calculate Portfolio Summary Data (but don't display yet)
            if st.session_state.portfolio:
                # Prepare data for the summary table
                summary_data = {
                    "Ticker": [s["ticker"] for s in st.session_state.portfolio],
                    "Company Name": [st.session_state.portfolio_info.get(s["ticker"], {}).get("company_name", "N/A") for s in st.session_state.portfolio],
                    "Min Weight (%)": [s["min"] for s in st.session_state.portfolio],
                    "Max Weight (%)": [s["max"] for s in st.session_state.portfolio],
                    "Currency": [st.session_state.portfolio_info.get(s["ticker"], {}).get("currency", "N/A") for s in st.session_state.portfolio],
                    "Beta": [],
                    "Actual Returns": [],
                    "Expected Return": []
                }
                
                # Extract risk-free rate as decimal
                try:
                    risk_free_decimal = float(risk_free_rate_str.replace('%', '')) / 100 if risk_free_rate_str != "N/A" else 0.04
                except:
                    risk_free_decimal = 0.04  # Default fallback
                
                # Extract market return as decimal
                try:
                    if market_return_str != "N/A":
                        market_return_decimal = float(market_return_str.replace('%', '')) / 100
                    else:
                        market_return_decimal = 0.08  # Default fallback
                except:
                    market_return_decimal = 0.08  # Default fallback
                
                # Fill in Beta, Actual Returns, and Expected Return for each ticker
                for ticker in summary_data["Ticker"]:
                    # Get Beta value
                    beta_value = "N/A"
                    if hasattr(st.session_state, 'risk_metrics_df') and not st.session_state.risk_metrics_df.empty:
                        beta_row = st.session_state.risk_metrics_df[st.session_state.risk_metrics_df['Ticker'] == ticker]
                        if not beta_row.empty:
                            beta_raw = beta_row['Beta'].iloc[0]
                            if not pd.isna(beta_raw):
                                beta_value = f"{beta_raw:.3f}"
                                beta_decimal = beta_raw
                            else:
                                beta_decimal = 1.0  # Default beta if N/A
                        else:
                            beta_decimal = 1.0  # Default beta if ticker not found
                    else:
                        beta_decimal = 1.0  # Default beta if no risk metrics
                    summary_data["Beta"].append(beta_value)
                    
                    # Calculate Actual Returns using the existing function
                    actual_return = "N/A"
                    if hasattr(st.session_state, 'portfolio_prices_usd_daily') and not st.session_state.portfolio_prices_usd_daily.empty:
                        if ticker in st.session_state.portfolio_prices_usd_daily.columns:
                            ticker_prices = st.session_state.portfolio_prices_usd_daily[ticker].dropna()
                            if not ticker_prices.empty and len(ticker_prices) >= 2:
                                try:
                                    annual_return = average_annualized_return(ticker_prices)
                                    if not np.isnan(annual_return):
                                        actual_return = f"{annual_return:.2%}"
                                    else:
                                        actual_return = "N/A"
                                except:
                                    actual_return = "N/A"
                    summary_data["Actual Returns"].append(actual_return)
                    
                    # Calculate Expected Return using CAPM formula
                    expected_return = "N/A"
                    if beta_value != "N/A" and risk_free_rate_str != "N/A" and market_return_str != "N/A":
                        try:
                            # CAPM: Expected Return = Risk-free rate + Beta * (Market return - Risk-free rate)
                            expected_return_decimal = risk_free_decimal + beta_decimal * (market_return_decimal - risk_free_decimal)
                            expected_return = f"{expected_return_decimal:.2%}"
                        except:
                            expected_return = "N/A"
                    summary_data["Expected Return"].append(expected_return)
                
                # Store the calculated summary data in session state
                st.session_state.portfolio_summary_data = pd.DataFrame(summary_data)
            else:
                st.session_state.portfolio_summary_data = None

            # ====== DISPLAY PORTFOLIO SUMMARY ======
            if hasattr(st.session_state, 'portfolio_summary_data') and st.session_state.portfolio_summary_data is not None:
                st.write("### Portfolio Summary")
                st.dataframe(st.session_state.portfolio_summary_data, use_container_width=True, hide_index=True)

            # Display Risk Metrics SECOND (even though they were calculated first)
            if hasattr(st.session_state, 'risk_metrics_df') and not st.session_state.risk_metrics_df.empty:
                st.write("### Risk Metrics")
                
                # Format the display
                risk_display_df = st.session_state.risk_metrics_df.copy()
                risk_display_df['Beta'] = risk_display_df['Beta'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
                risk_display_df['Volatility (Annualized)'] = risk_display_df['Volatility (Annualized)'].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")
                risk_display_df['Correlation with Benchmark'] = risk_display_df['Correlation with Benchmark'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
                risk_display_df['Sharpe Ratio'] = risk_display_df['Sharpe Ratio'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
                
                st.dataframe(risk_display_df, use_container_width=True, hide_index=True)

                # Display benchmark info
                if hasattr(st.session_state, 'benchmark_volatility') and hasattr(st.session_state, 'selected_benchmark_name'):
                    benchmark_vol = st.session_state.benchmark_volatility
                    benchmark_name = st.session_state.selected_benchmark_name
                    
                    # If user selected "Specify ticker", show the actual ticker instead
                    if benchmark_name == "Specify ticker" and hasattr(st.session_state, 'benchmark_ticker'):
                        benchmark_name = st.session_state.benchmark_ticker                       

                    # Display benchmark volatility
                    st.write(f"**Benchmark ({benchmark_name}) Volatility (Full Period):** {benchmark_vol:.2%}" if not np.isnan(benchmark_vol) else "**Benchmark Volatility:** N/A")
                    
                    # Calculate and display benchmark Sharpe ratio
                    if market_return_str != "N/A" and risk_free_rate_str != "N/A" and not np.isnan(benchmark_vol):
                        try:
                            market_return_decimal = float(market_return_str.replace('%', '')) / 100
                            risk_free_decimal = float(risk_free_rate_str.replace('%', '')) / 100
                            benchmark_sharpe = (market_return_decimal - risk_free_decimal) / benchmark_vol
                            st.write(f"**Benchmark ({benchmark_name}) Sharpe Ratio:** {benchmark_sharpe:.3f}")
                        except:
                            st.write(f"**Benchmark ({benchmark_name}) Sharpe Ratio:** N/A")
                    else:
                        st.write(f"**Benchmark ({benchmark_name}) Sharpe Ratio:** N/A")                
                
                # Show interpretation
                st.write("### Beta Interpretation")
                st.write("- **Beta > 1:** Stock is more volatile than the benchmark")
                st.write("- **Beta = 1:** Stock moves with the benchmark") 
                st.write("- **Beta < 1:** Stock is less volatile than the benchmark")
                st.write("- **Beta < 0:** Stock moves opposite to the benchmark")
                    
            else:
                if len(tickers) < 2:
                    st.info("Add at least 2 tickers to calculate risk metrics.")
                else:
                    st.info("No returns data available for risk metric calculations.")

            # At the end, show success message
            st.success("✅ Portfolio analysis complete!")

    # Show previous results if available (INSIDE the tickers check)
    elif hasattr(st.session_state, 'portfolio_summary_data') and st.session_state.portfolio_summary_data is not None:
        st.write("### Previous Results")
        st.info("Click the button above to refresh with current portfolio settings.")
        
        st.write("#### Last Portfolio Summary")
        st.dataframe(st.session_state.portfolio_summary_data, use_container_width=True, hide_index=True)
        
        if hasattr(st.session_state, 'risk_metrics_df') and not st.session_state.risk_metrics_df.empty:
            st.write("#### Last Risk Metrics")
            st.dataframe(st.session_state.risk_metrics_df, use_container_width=True, hide_index=True)

else:  # This runs when there are NO tickers
    if not tickers:
        st.info("👆 Add some tickers to your portfolio to get started!")

# === Inline Correlation Matrix (Page 2) ===
def render_correlation_matrix_inline():
    import numpy as np
    import plotly.express as px

    # 1) Only render when data exists — otherwise do nothing (no messages)
    returns_df = None
    period_label = None

    # Prefer selected-period returns, else fall back to 5-year
    if ("portfolio_returns_selected_period" in st.session_state
        and isinstance(st.session_state.portfolio_returns_selected_period, pd.DataFrame)
        and not st.session_state.portfolio_returns_selected_period.empty):
        returns_df = st.session_state.portfolio_returns_selected_period.copy()
        period_label = "Selected Period"
    elif ("five_year_returns_df" in st.session_state
          and isinstance(st.session_state.five_year_returns_df, pd.DataFrame)
          and not st.session_state.five_year_returns_df.empty):
        returns_df = st.session_state.five_year_returns_df.copy()
        period_label = "Past 5 Years"

    if returns_df is None:
        return  # silently skip until analysis has been run

    # 2) Keep only tickers currently in the portfolio
    current_tickers = [s["ticker"] for s in st.session_state.get("portfolio", [])]
    if not current_tickers:
        return

    cols = [t for t in returns_df.columns if t in current_tickers]
    if len(cols) < 2:
        return

    returns_df = returns_df[cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if returns_df.shape[0] < 2:
        return

    # 3) Compute correlation & covariance from Page 2 returns
    corr_df = returns_df.corr()
    cov_df = returns_df.cov()

    # 4) Inline title + heatmap (match Page 3/4 style)
    st.subheader(f"📊 Correlation Matrix ({period_label})")

    corr_plot = corr_df.clip(-1.0, 1.0)
    fig = px.imshow(
        corr_plot,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",  # match Page 4 palette
        zmin=-1, zmax=1,
        labels=dict(color="Correlation")
    )

    # match Page 3: slanted x-axis labels + readable values
    fig.update_traces(texttemplate="%{z:.2f}")
    fig.update_xaxes(side="bottom", tickangle=45)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticks="outside"
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("−1 = move opposite · 0 = weak/none · +1 = move together")

    # 5) Optional downloads for transparency (still inline)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download **Correlation Matrix** (CSV)",
            data=corr_df.to_csv(index=True).encode("utf-8"),
            file_name="correlation_matrix.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            "⬇️ Download **Covariance Matrix** (CSV)",
            data=cov_df.to_csv(index=True).encode("utf-8"),
            file_name="covariance_matrix.csv",
            mime="text/csv",
            help="Raw covariance computed from this page’s returns"
        )

# Call it at the bottom of Page 2
render_correlation_matrix_inline()

# === Data Exports: Prices & FX (session-backed, stable on rerun) ===
st.write("---")
st.subheader("📥 Download Data: Prices & FX")

def _dl_df(label, df, fname, key):
    import pandas as pd
    if df is not None and isinstance(df, (pd.DataFrame, pd.Series)) and not df.empty:
        st.download_button(
            f"⬇️ {label}",
            data=df.to_csv(index=True).encode("utf-8"),
            file_name=fname,
            mime="text/csv",
            key=key  # stable per-button key so reruns don't confuse Streamlit
        )

# Build from session only

fx_dict = st.session_state.get("fx_data", {})
if fx_dict:
    for fx_ticker, fx_df in fx_dict.items():
        _dl_df(
            f"fx_data ({fx_ticker})",
            fx_df,
            f"fx_data_{fx_ticker}.csv",
            key=f"dl_fx_{fx_ticker}"
        )

_dl_df("Benchmark Prices (Daily, Selected Period, Local currency)",
       st.session_state.get("data_all"), "data_all.csv", "dl_data_all")

_dl_df("Benchmark Prices (Daily, Selected Period, USD)",
       st.session_state.get("benchmark_prices_usd"), "benchmark_prices_usd.csv", "dl_benchmark_prices_usd")

_dl_df("Benchmark Prices (Daily, 5-year, USD)",
       st.session_state.get("benchmark_prices_5y_usd"), "benchmark_prices_5y_usd.csv", "dl_benchmark_prices_5y_usd")

_dl_df("Benchmark Returns (Daily, 5-year)",
       st.session_state.get("benchmark_returns_5y"), "benchmark_returns_5y.csv", "dl_benchmark_returns_5y")

_dl_df("Portfolio Prices (Daily, Selected Period, Local currency)",
       st.session_state.get("portfolio_prices_df"), "portfolio_prices_local.csv", "dl_portfolio_prices_df")

_dl_df("Portfolio Prices (Daily, Selected Period, USD)",
       st.session_state.get("portfolio_prices_usd_daily"), "portfolio_prices_usd_daily.csv", "dl_pp_usd_daily")

_dl_df("Portfolio Prices (Daily, 5-year, Local currency)",
       st.session_state.get("prices_5y_df"), "prices_5y_local_5y.csv", "dl_prices_5y_df")

_dl_df("Portfolio Prices (Daily, 5-year, USD)",
       st.session_state.get("prices_5y_usd_df"), "prices_5y_usd_5y.csv", "dl_prices_5y_usd_df")

_dl_df("Portfolio Returns (Daily, Selected Period)",
       st.session_state.get("portfolio_returns_selected_period"), "portfolio_returns_selected_period.csv", "dl_portfolio_returns_selected")

_dl_df("Portfolio Returns (Daily, 5-year)",
       st.session_state.get("five_year_returns_df"), "five_year_returns.csv", "dl_five_year_returns_df")




