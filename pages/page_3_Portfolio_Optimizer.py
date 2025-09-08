import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pypfopt import EfficientFrontier, risk_models, expected_returns, black_litterman, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import objective_functions
import warnings
warnings.filterwarnings('ignore')
from pypfopt import plotting

st.set_page_config(page_title="Portfolio Optimizer", page_icon="🚀", layout="wide")

def initialize_optimizer_session_state():
    """Initialize session state variables for optimizer"""
    if "stabilize_covariance" not in st.session_state:
        st.session_state.stabilize_covariance = True
    if "use_black_litterman" not in st.session_state:
        st.session_state.use_black_litterman = False
    if "market_views" not in st.session_state:
        st.session_state.market_views = {}
    if "optimization_objectives" not in st.session_state:
        st.session_state.optimization_objectives = {
            "max_sharpe": False,
            "min_volatility": False,
            "max_return_fixed_risk": False,
            "min_risk_fixed_return": False
        }
    if "target_risk" not in st.session_state:
        st.session_state.target_risk = 0.15
    if "target_return" not in st.session_state:
        st.session_state.target_return = 0.10

def check_data_availability():
    """Check if required data is available from the asset analyzer page"""
    required_data = [
        'portfolio',
        'portfolio_prices_usd_daily',
        'portfolio_summary_data'  # This contains the actual returns we already calculated
    ]
    
    missing_data = []
    for data_key in required_data:
        if data_key not in st.session_state or (
            hasattr(st.session_state, data_key) and 
            isinstance(getattr(st.session_state, data_key), pd.DataFrame) and 
            getattr(st.session_state, data_key).empty
        ):
            missing_data.append(data_key)
    
    return len(missing_data) == 0, missing_data

def create_efficient_frontier_plot(ef, tickers, weights_dict=None):
    """Create an interactive efficient frontier plot"""
    # Create the plot using matplotlib first (for professor's function)
    import matplotlib.pyplot as plt
    fig_mpl, ax = plt.subplots(figsize=(10, 6))

    # Use professor's robust frontier generation
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False, n_points=200)

    # Convert matplotlib to plotly for your interface
    # (We'll extract the frontier data from matplotlib and put it in plotly)
    frontier_line = ax.lines[0]  # Get the frontier line
    frontier_x = frontier_line.get_xdata()
    frontier_y = frontier_line.get_ydata()
    plt.close(fig_mpl)  # Close matplotlib figure

    # Create plotly figure
    fig = go.Figure()

    # Add efficient frontier using professor's data
    fig.add_trace(go.Scatter(
        x=frontier_x,
        y=frontier_y,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='blue', width=3)
    ))
    
    
    # Add Monte Carlo simulated portfolios
    num_portfolios = 1000
    mc_rets = []
    mc_risks = []
    
    for _ in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(weights * ef.expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(ef.cov_matrix, weights)))
        
        mc_rets.append(portfolio_return)
        mc_risks.append(portfolio_risk)
    
    # Add Monte Carlo points
    fig.add_trace(go.Scatter(
        x=mc_risks,
        y=mc_rets,
        mode='markers',
        name='Random Portfolios',
        marker=dict(
            size=4,
            color=mc_risks,
            colorscale='Viridis',
            opacity=0.6
        )
    ))
    
    # Add optimized portfolios if provided
    if weights_dict:
        for name, weights in weights_dict.items():
            if weights is not None:
                portfolio_return = np.sum(weights * ef.expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(ef.cov_matrix, weights)))
                
                fig.add_trace(go.Scatter(
                    x=[portfolio_risk],
                    y=[portfolio_return],
                    mode='markers+text',
                    name=name,
                    text=[name],
                    textposition="top center",
                    marker=dict(size=12, symbol='star'),
                    showlegend=True
                ))
    
    fig.update_layout(
        title='Efficient Frontier with Optimized Portfolios',
        xaxis_title='Risk (Volatility)',
        yaxis_title='Expected Return',
        hovermode='closest',
        height=600
    )
    
    return fig

def average_annualized_return(price_series):
    """Custom annualized return calculation from the Asset Analyzer"""
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

def clean_and_align_prices(prices_df):
    """Professional cleaning for both tables: 0→NaN, then limited forward-fill"""
    # Step 1: Convert 0s to NaN (market closures/holidays)
    clean_prices = prices_df.replace(0, np.nan)
    
    # Step 2: Forward-fill with 5-day limit (avoid overfilling suspensions)
    aligned_prices = clean_prices.ffill(limit=5)
    
    return aligned_prices


def display_portfolio_allocation(weights, tickers, title, portfolio_groups=None):
    """Display portfolio allocation with pie chart and table"""
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Pie chart
        weights_percent = [w * 100 for w in weights if w > 0.001]  # Filter out tiny allocations
        labels = [tickers[i] for i, w in enumerate(weights) if w > 0.001]
        
        if weights_percent:
            fig_pie = px.pie(
                values=weights_percent,
                names=labels,
                title=f"{title} - Asset Allocation"
            )
            st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_chart_{title.replace(' ', '_').replace('@', 'at') if title else 'unnamed'}")
    
    with col2:
        # Allocation table
        allocation_data = []
        for i, ticker in enumerate(tickers):
            if weights[i] > 0.001:  # Only show meaningful allocations
                allocation_data.append({
                    'Ticker': ticker,
                    'Weight (%)': f"{weights[i]*100:.2f}%"
                })
        
        if allocation_data:
            allocation_df = pd.DataFrame(allocation_data)
            st.dataframe(allocation_df, use_container_width=True, hide_index=True)
    
    # Group allocation if groups are enabled
    if portfolio_groups and st.session_state.get('enable_asset_groups', False):
        st.write("**Group Allocation:**")
        group_weights = {}
        
        for i, ticker in enumerate(tickers):
            if weights[i] > 0.001:
                # Find the group for this ticker
                ticker_group = None
                for stock in st.session_state.portfolio:
                    if stock['ticker'] == ticker:
                        ticker_group = stock.get('group', 'No Group')
                        break
                
                if ticker_group and ticker_group != 'No Group':
                    if ticker_group not in group_weights:
                        group_weights[ticker_group] = 0
                    group_weights[ticker_group] += weights[i]
        
        if group_weights:
            group_data = []
            for group, weight in group_weights.items():
                group_name = st.session_state.asset_groups.get(group, {}).get('name', group)
                display_name = group_name if group_name else group
                group_data.append({
                    'Group': display_name,
                    'Weight (%)': f"{weight*100:.2f}%"
                })
            
            group_df = pd.DataFrame(group_data)
            st.dataframe(group_df, use_container_width=True, hide_index=True)

def get_market_cap_weights(tickers):
    """Get market cap weights for tickers with USD conversion"""
    
    # Initialize storage for market cap data
    market_cap_data = {}
    
    if st.session_state.get('bl_weight_method') == "Equal Weights (simplified)":
        # Equal weights
        weights = np.array([1.0/len(tickers)] * len(tickers))
        st.write("📊 Using equal weights as market cap proxy")
        
        # Store equal weight info
        for ticker in tickers:
            market_cap_data[ticker] = "Equal weight"
        
    elif st.session_state.get('bl_weight_method') == "Manual Input":
        # User-defined weights
        manual_weights = st.session_state.get('bl_manual_weights', {})
        weights = np.array([manual_weights.get(ticker, 1.0/len(tickers)) for ticker in tickers])
        st.write("📊 Using manually entered market cap weights")
        
        # Store manual weight info
        for ticker in tickers:
            market_cap_data[ticker] = "Manual input"
        
    else:  # "Fetch Market Cap Data"
        st.write("📊 Attempting to fetch market cap data...")
        market_caps = []
        
        # Get currency mapping from portfolio info (same as Portfolio Analyzer)
        currency_map = {
            s["ticker"]: st.session_state.portfolio_info.get(s["ticker"], {}).get('currency', 'USD')
            for s in st.session_state.portfolio
        }
        
        for ticker in tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                market_cap = info.get('marketCap', None)
                currency = currency_map.get(ticker, 'USD')
                
                if market_cap and market_cap > 0:
                    # Convert to USD if needed (same logic as Portfolio Analyzer)
                    market_cap_usd = market_cap
                    
                    if currency not in ["USD", "N/A", None]:
                        fx_ticker = f"{currency}USD=X"
                        
                        try:
                            # Get latest FX rate (similar to Portfolio Analyzer approach)
                            fx_data = yf.download(
                                fx_ticker,
                                period="5d",  # Get recent data
                                progress=False,
                                auto_adjust=True
                            )
                            
                            if not fx_data.empty and 'Close' in fx_data.columns:
                                fx_rates_raw = fx_data['Close']
                                
                                # Handle DataFrame vs Series
                                if isinstance(fx_rates_raw, pd.DataFrame):
                                    fx_rate = fx_rates_raw.iloc[-1, 0]  # Latest rate
                                elif isinstance(fx_rates_raw, pd.Series):
                                    fx_rate = fx_rates_raw.iloc[-1]  # Latest rate
                                else:
                                    fx_rate = 1.0  # Fallback
                                
                                # Convert to numeric and apply conversion
                                fx_rate = pd.to_numeric(fx_rate, errors='coerce')
                                if not pd.isna(fx_rate) and fx_rate > 0:
                                    market_cap_usd = market_cap * fx_rate
                                    st.write(f"✅ {ticker}: {currency} {market_cap:,.0f} → ${market_cap_usd:,.0f} (FX: {fx_rate:.4f})")
                                else:
                                    st.warning(f"⚠️ {ticker}: Invalid FX rate, using original {currency} value")
                                    st.write(f"✅ {ticker}: {currency} {market_cap:,.0f} (no conversion)")
                            else:
                                st.warning(f"⚠️ {ticker}: No FX data for {fx_ticker}, using original {currency} value")
                                st.write(f"✅ {ticker}: {currency} {market_cap:,.0f} (no conversion)")
                                
                        except Exception as fx_error:
                            st.warning(f"⚠️ {ticker}: FX conversion failed ({str(fx_error)}), using original {currency} value")
                            st.write(f"✅ {ticker}: {currency} {market_cap:,.0f} (no conversion)")
                    else:
                        # Already in USD or unknown currency
                        st.write(f"✅ {ticker}: ${market_cap_usd:,.0f}")
                    
                    market_caps.append(market_cap_usd)
                    market_cap_data[ticker] = market_cap_usd
                    
                else:
                    st.warning(f"⚠️ {ticker}: Market cap not found, using equal weight")
                    market_caps.append(1.0)  # Fallback
                    market_cap_data[ticker] = "Not found - equal weight"
                    
            except Exception as e:
                st.warning(f"⚠️ {ticker}: Error fetching data ({str(e)}), using equal weight")
                market_caps.append(1.0)  # Fallback
                market_cap_data[ticker] = f"Error - equal weight"
        
        # Convert to weights
        total_market_cap = sum(market_caps)
        weights = np.array([cap/total_market_cap for cap in market_caps])
        
        # Display total market cap in USD
        if total_market_cap > 1:  # Only if we have real market cap data
            st.write(f"**Total Portfolio Market Cap:** ${total_market_cap:,.0f}")
    
    # Store market cap data for the table
    st.session_state.bl_market_caps = market_cap_data
    
    # Validate weights
    if abs(weights.sum() - 1.0) > 0.01:
        st.warning(f"Market cap weights sum to {weights.sum():.3f}, normalizing to 1.0")
        weights = weights / weights.sum()
    
    return weights

def calculate_market_implied_returns(cov_matrix, market_weights, delta):
    """Calculate market-implied equilibrium returns: Π = δ·Σ·w_mkt"""
    pi_implied = delta * np.dot(cov_matrix, market_weights)
    return pd.Series(pi_implied, index=cov_matrix.index)

# Main page content starts here
st.title('🚀 Portfolio Optimizer')

# Initialize session state
initialize_optimizer_session_state()

# Check if required data is available
data_available, missing_data = check_data_availability()

if not data_available:
    st.error("⚠️ Required data not found!")
    st.write("Please complete the following steps first:")
    st.write("1. Go to the **📊 Portfolio Asset Analyzer** page")
    st.write("2. Add at least 2 tickers to your portfolio")
    st.write("3. Click '🚀 Get Portfolio Summary and Risk Metrics!'")
    st.write("4. Return to this optimization page")
    st.stop()

# Get data from session state
tickers = [s["ticker"] for s in st.session_state.portfolio]

if len(tickers) < 2:
    st.error("❌ Please add at least 2 tickers to optimize a portfolio")
    st.stop()

st.success(f"✅ Ready to optimize portfolio with {len(tickers)} assets: {', '.join(tickers)}")

# Configuration Options
st.subheader("Configuration Options")

col1, col2 = st.columns(2)

with col1:
    st.session_state.stabilize_covariance = st.checkbox(
        "Stabilize covariance matrix (recommended for Black-Litterman, shorter time frames or many assets)",
        value=st.session_state.stabilize_covariance,
        key="stabilize_cov_checkbox"
    )

with col2:
    st.session_state.use_black_litterman = st.checkbox(
        "Apply your own market views (Black-Litterman Model)",
        value=st.session_state.use_black_litterman,
        key="bl_checkbox"
    )

# Black-Litterman Market Views
if st.session_state.use_black_litterman:
    st.subheader("Black-Litterman Model Configuration")
    
    # Prior selection (new addition)
    st.write("**Step 1: Select Market Prior**")
    prior_options = [
        "Historical Returns (from Asset Analyzer)",
        "Market-Implied Equilibrium Returns (textbook Black-Litterman)"
    ]
    
    selected_prior = st.radio(
        "Choose the baseline prior returns:",
        prior_options,
        key="bl_prior_selection",
        help="Historical returns use your actual calculated returns. Market-implied returns use market cap weights and risk aversion."
    )
    
    # Store the selection
    st.session_state.bl_use_market_implied = (selected_prior == prior_options[1])
    
    # Show additional options for market-implied priors
    if st.session_state.bl_use_market_implied:
        st.write("**Market-Implied Prior Settings:**")
        
        col_delta, col_method = st.columns(2)
        
        with col_delta:
            # Risk aversion coefficient
            delta = st.number_input(
                "Risk Aversion Coefficient (δ)",
                min_value=1.0,
                max_value=10.0,
                value=2.5,
                step=0.1,
                key="bl_delta",
                help="Represents market risk aversion. Common range: 2.5-4.0"
            )
            #st.session_state.bl_delta = delta
        
        with col_method:
            # Market cap weight method
            weight_method = st.selectbox(
                "Market Cap Weight Method",
                ["Equal Weights (simplified)", "Fetch Market Cap Data", "Manual Input"],
                key="bl_weight_method",
                help="How to determine market capitalization weights"
            )
            #st.session_state.bl_weight_method = weight_method
        
        # Handle different weight methods
        if weight_method == "Equal Weights (simplified)":
            st.info("💡 Using equal weights as a simplified approximation of market cap weights.")
        
        elif weight_method == "Fetch Market Cap Data":
            st.info("📊 Will attempt to fetch market cap data from yfinance.")
            
        elif weight_method == "Manual Input":
            st.write("**Enter Market Cap Weights (must sum to 1.0):**")
            
            manual_weights = {}
            weight_cols = st.columns(min(len(tickers), 3))  # Max 3 columns
            
            for i, ticker in enumerate(tickers):
                with weight_cols[i % len(weight_cols)]:
                    weight = st.number_input(
                        f"{ticker} weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0/len(tickers),  # Default to equal weights
                        step=0.01,
                        key=f"manual_weight_{ticker}"
                    )
                    manual_weights[ticker] = weight
            
            # Validate weights sum to 1
            total_weight = sum(manual_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                st.error(f"⚠️ Weights sum to {total_weight:.3f}, must sum to 1.0")
            else:
                st.success(f"✅ Weights sum to {total_weight:.3f}")
                st.session_state.bl_manual_weights = manual_weights
    
    else:
        st.info("💡 Using historical returns from your Asset Analyzer as market priors.")
    
    st.write("---")
    st.write("**Step 2: Enter Your Market Views**")
    st.write("Enter your expected annual returns and confidence levels for each asset:")
    
    # Rest of your existing market views code...
    # Create headers for the market views table
    col_header1, col_header2, col_header3 = st.columns([2, 2, 2])
    with col_header1:
        st.write("**Ticker**")
    with col_header2:
        st.write("**Market View (% annual return)**")
    with col_header3:
        st.write("**Confidence (%)**")
    
    # Create market views table
    view_data = []
    for ticker in tickers:
        # Get current view or set default
        current_view = st.session_state.market_views.get(ticker, {'return': 10.0, 'confidence': 50})
        
        col_ticker, col_return, col_confidence = st.columns([2, 2, 2])
        
        with col_ticker:
            st.write(f"**{ticker}**")
        
        with col_return:
            view_return = st.number_input(
                "Market view (% annual return)",
                min_value=-50.0,
                max_value=100.0,
                value=current_view['return'],
                step=0.5,
                key=f"view_return_{ticker}",
                label_visibility="collapsed"
            )
        
        with col_confidence:
            confidence = st.slider(
                "Confidence (%)",
                min_value=1,
                max_value=100,
                value=current_view['confidence'],
                key=f"confidence_{ticker}",
                label_visibility="collapsed"
            )
        
        # Store in session state
        st.session_state.market_views[ticker] = {
            'return': view_return,
            'confidence': confidence
        }
        
        view_data.append({
            'Ticker': ticker,
            'Market View (%)': f"{view_return:.1f}%",
            'Confidence (%)': f"{confidence}%"
        })
    
    # Display summary table
    if view_data:
        st.write("**Summary of Market Views:**")
        view_df = pd.DataFrame(view_data)
        st.dataframe(view_df, use_container_width=True, hide_index=True)

# Optimization Objectives
# Optimization Objectives
st.subheader("Please select your optimization objective(s):")

objective_col1, objective_col2 = st.columns(2)

with objective_col1:
    st.session_state.optimization_objectives["max_sharpe"] = st.checkbox(
        "Maximize Sharpe Ratio",
        value=st.session_state.optimization_objectives["max_sharpe"],
        key="max_sharpe_checkbox"
    )
    
    st.session_state.optimization_objectives["min_volatility"] = st.checkbox(
        "Minimize Volatility",
        value=st.session_state.optimization_objectives["min_volatility"],
        key="min_vol_checkbox"
    )

with objective_col2:
    st.session_state.optimization_objectives["max_return_fixed_risk"] = st.checkbox(
        "Maximize Return @ Fixed Risk",
        value=st.session_state.optimization_objectives["max_return_fixed_risk"],
        key="max_ret_checkbox"
    )
    
    if st.session_state.optimization_objectives["max_return_fixed_risk"]:
        st.write("**Select your risk tolerance:**")
        
        # Risk level definitions
        risk_levels = {
            1: {"name": "Conservative", "range": "0-15%", "min": 0, "max": 15, "default": 7.5},
            2: {"name": "Moderate Conservative", "range": "15-25%", "min": 15, "max": 25, "default": 20.0},
            3: {"name": "Moderate", "range": "25-40%", "min": 25, "max": 40, "default": 32.5},
            4: {"name": "Moderate Aggressive", "range": "40-60%", "min": 40, "max": 60, "default": 50.0},
            5: {"name": "Aggressive", "range": "60-120%", "min": 60, "max": 120, "default": 90.0}
        }
        
        # Risk level slider
        risk_level = st.slider(
            "Risk Level",
            min_value=1,
            max_value=5,
            value=3,  # Default to Moderate
            step=1,
            format="%d",
            key="risk_level_slider"
        )
        
        # Display selected risk level info
        selected_risk = risk_levels[risk_level]
        st.info(f"**{selected_risk['name']}** - Typical volatility range: {selected_risk['range']}")
        
        # Choice between range or manual input
        input_method = st.radio(
            "How would you like to set your risk level?",
            ["Use recommended range", "Enter specific volatility"],
            key="risk_input_method"
        )
        

        if input_method == "Use recommended range":
            # Use the default value for the selected risk level  
            target_risk_percent = selected_risk['default']
            #st.success(f"Will optimize for best Sharpe ratio within {selected_risk['range']} volatility range")

            # Store range optimization parameters
            st.session_state.optimization_type = "max_sharpe_in_range"
            st.session_state.volatility_min = selected_risk['min'] / 100
            st.session_state.volatility_max = selected_risk['max'] / 100
            st.session_state.target_risk = target_risk_percent / 100  # Keep for display
        else:
            # Manual input in percentage
            target_risk_percent = st.number_input(
                "Enter your target volatility (%):",
                min_value=5.0,
                max_value=50.0,
                value=selected_risk['default'],
                step=0.5,
                format="%.1f",
                key="manual_risk_input"
            )
            
            # Store fixed risk optimization parameters  
            st.session_state.optimization_type = "max_return_fixed_risk"
            st.session_state.target_risk = target_risk_percent / 100

        if st.session_state.get('optimization_type') == "max_sharpe_in_range":
            st.write(f"**Selected range:** {selected_risk['range']} volatility")
        else:
            st.write(f"**Selected volatility:** {target_risk_percent}%")
    
    st.session_state.optimization_objectives["min_risk_fixed_return"] = st.checkbox(
        "Minimize Risk @ Fixed Return",
        value=st.session_state.optimization_objectives["min_risk_fixed_return"],
        key="min_risk_checkbox"
    )

    if st.session_state.optimization_objectives["min_risk_fixed_return"]:
        target_return_percent = st.number_input(
            "Enter your target return (%):",
            min_value=1.0,
            max_value=200.0,
            value=st.session_state.target_return * 100,  # Convert stored decimal to percentage for display
            step=0.5,
            format="%.1f",
            key="target_return_input"
        )
        # Convert percentage back to decimal for storage
        st.session_state.target_return = target_return_percent / 100

# Check if at least one objective is selected
objectives_selected = any(st.session_state.optimization_objectives.values())

if not objectives_selected:
    st.warning("⚠️ Please select at least one optimization objective.")
    st.stop()

# Optimize Button
st.write("---")
if st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True):

    with st.status("🚀 Optimizing portfolio...", expanded=True) as status:
        try:
            # Use pre-cleaned data from Asset Analyzer instead of re-processing
            if 'portfolio_summary_data' not in st.session_state or st.session_state.portfolio_summary_data is None:
                st.error("❌ No portfolio summary data found. Please run the Asset Analyzer first.")
                st.stop()

            summary_data = st.session_state.portfolio_summary_data

            # Use already-cleaned price data (no re-cleaning needed!)
            price_data = st.session_state.portfolio_prices_usd_daily[tickers]

            st.write("**Using Pre-Cleaned Data from Asset Analyzer:**")
            st.write(f"Date range: {price_data.index[0].strftime('%Y-%m-%d')} to {price_data.index[-1].strftime('%Y-%m-%d')}")

            # Extract the actual returns (already calculated in Asset Analyzer)
            st.write("**Actual Returns (from Asset Analyzer):**")
            mu = pd.Series(index=tickers, dtype=float)

            for ticker in tickers:
                # Find the actual return for this ticker from your summary data
                ticker_row = summary_data[summary_data['Ticker'] == ticker]
                if not ticker_row.empty:
                    actual_return_str = ticker_row['Actual Returns'].iloc[0]
                    if actual_return_str != "N/A":
                        # Convert percentage string back to decimal (e.g., "15.23%" -> 0.1523)
                        actual_return = float(actual_return_str.replace('%', '')) / 100
                        mu[ticker] = actual_return
                        st.write(f"- {ticker}: {actual_return:.2%} (from Asset Analyzer)")
                    else:
                        st.warning(f"⚠️ {ticker}: No actual return calculated in Asset Analyzer")
                        mu[ticker] = 0.0
                else:
                    st.warning(f"⚠️ {ticker}: Not found in portfolio summary")
                    mu[ticker] = 0.0

            # Use selected period returns data (NOT 5-year data) for covariance calculation
            if hasattr(st.session_state, 'portfolio_returns_selected_period') and not st.session_state.portfolio_returns_selected_period.empty:
                # Use the selected period returns dataframe from Asset Analyzer
                returns_data = st.session_state.portfolio_returns_selected_period[tickers].copy()
                st.write(f"✅ Using selected period returns data: {returns_data.shape}")
                st.write(f"Date range: {returns_data.index[0].strftime('%Y-%m-%d')} to {returns_data.index[-1].strftime('%Y-%m-%d')}")
                
                # Only minimal additional cleaning for covariance matrix stability
                # Remove any rows where ALL returns are zero (market holidays) - already done but double-check
                all_zero_mask = (returns_data == 0).all(axis=1)
                zero_days = all_zero_mask.sum()
                if zero_days > 0:
                    st.write(f"Removing {zero_days} additional days where all returns are zero")
                    returns_data = returns_data[~all_zero_mask]
                
                # Add tiny random noise to prevent identical returns (only if needed for covariance stability)
                if returns_data.nunique().min() < len(returns_data) * 0.1:  # Only if many identical values
                    st.write("Adding minimal noise for covariance matrix stability")
                    np.random.seed(42)  # For reproducibility
                    noise = np.random.normal(0, 1e-8, returns_data.shape)
                    returns_data = returns_data + noise
                
                st.write(f"Final selected period returns data for covariance: {returns_data.shape}")
                
            else:
                # Fallback: calculate returns from selected period price data
                st.warning("Selected period returns not available, calculating from selected period prices...")
                
                # Use the selected period prices (NOT 5-year prices)
                price_data = st.session_state.portfolio_prices_usd_daily[tickers]
                returns_data = price_data.pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)
                
                # Remove market holiday zeros
                all_zero_mask = (returns_data == 0).all(axis=1)
                if all_zero_mask.sum() > 0:
                    returns_data = returns_data[~all_zero_mask]
                
                st.write(f"Calculated returns from selected period prices: {returns_data.shape}")

            # Validation
            if 'returns_data' not in locals():
                st.error("❌ Returns data not properly initialized")
                st.stop()

            if returns_data.empty:
                st.error("❌ No valid daily returns data for covariance calculation")
                st.stop()

            if len(returns_data) < 30:
                st.warning(f"⚠️ Limited data: Only {len(returns_data)} days available. Covariance estimates may be unreliable.")

            st.success("✅ Successfully using selected period returns data for optimization")
            

            # Ensure returns_data is always defined for covariance calculation
            if 'returns_data' not in locals():
                st.error("❌ Returns data not properly initialized")
                st.stop()

            if returns_data.empty:
                st.error("❌ No valid daily returns data for covariance calculation")
                st.stop()

            if len(returns_data) < 30:
                st.warning(f"⚠️ Limited data: Only {len(returns_data)} days available. Covariance estimates may be unreliable.")

            st.success("✅ Successfully using pre-processed data from Asset Analyzer")
            
            # Calculate covariance matrix (we only need this, since expected returns are already calculated)
            st.write("**Calculating Covariance Matrix:**")
            
            try:
                # Replace the Black-Litterman section in your optimization code with this:

                if st.session_state.use_black_litterman:
                    # Black-Litterman approach
                    st.write("Using Black-Litterman model...")
                    
                    # Determine which prior to use
                    if st.session_state.get('bl_use_market_implied', False):
                        # Use market-implied equilibrium returns
                        st.write("**Using market-implied equilibrium returns as priors**")
                        
                        # FIRST: Calculate base covariance matrix (needed for market-implied returns)
                        price_data = st.session_state.portfolio_prices_usd_daily[tickers]
                        if st.session_state.stabilize_covariance:
                            S = risk_models.CovarianceShrinkage(price_data).ledoit_wolf()
                        else:
                            S = risk_models.sample_cov(price_data)
                        
                        # THEN: Get market cap weights and calculate market-implied returns
                        market_weights = get_market_cap_weights(tickers)
                        delta = st.session_state.get('bl_delta', 2.5)
                        
                        # Calculate market-implied returns: Π = δ·Σ·w_mkt
                        mu_market = calculate_market_implied_returns(S, market_weights, delta)

                        # Store data for the detailed table
                        st.session_state.bl_table_data = {
                            'tickers': tickers,
                            'prior_type': 'market_implied',
                            'market_caps': getattr(st.session_state, 'bl_market_caps', {}),  # Will be populated by get_market_cap_weights
                            'market_weights': dict(zip(tickers, market_weights)),
                            'market_implied_returns': dict(zip(tickers, mu_market)),
                            'historical_returns': None,  # Not used when market-implied
                            'user_views': {ticker: st.session_state.market_views.get(ticker, {}) for ticker in tickers},
                            'final_returns': None  # Will be set after BL calculation
                        }
                        
                        st.write("**Market-Implied Equilibrium Returns:**")
                        for ticker in tickers:
                            st.write(f"- {ticker}: {mu_market[ticker]:.2%}")
                            
                    else:
                        # Use historical returns from Asset Analyzer
                        st.write("**Using historical returns from Asset Analyzer as priors**")
                        mu_market = mu.copy()
                        
                        st.write("**Historical Returns (used as priors):**")
                        for ticker in tickers:
                            st.write(f"- {ticker}: {mu_market[ticker]:.2%}")

                        # Store data for the detailed table (historical returns case)
                        st.session_state.bl_table_data = {
                            'tickers': tickers,
                            'prior_type': 'historical',
                            'market_caps': {},  # Not applicable
                            'market_weights': {},  # Not applicable  
                            'market_implied_returns': {},  # Not applicable
                            'historical_returns': dict(zip(tickers, mu_market)),
                            'user_views': {ticker: st.session_state.market_views.get(ticker, {}) for ticker in tickers},
                            'final_returns': None  # Will be set after BL calculation
                        }
                    
                    # Create views dictionary from user inputs (same as before)
                    views = {}
                    confidences = []

                    for ticker in tickers:
                        view_data = st.session_state.market_views.get(ticker, {'return': 10.0, 'confidence': 50})
                        views[ticker] = view_data['return'] / 100  # Convert to decimal
                        
                        # Convert confidence percentage to proper omega (uncertainty) values
                        confidence_pct = view_data['confidence']  # 1-100
                        omega_value = (101 - confidence_pct) / 100  # Inverts the scale
                        confidences.append(omega_value)
                    
                    # Calculate base covariance matrix from price data
                    if st.session_state.stabilize_covariance:
                        S = risk_models.CovarianceShrinkage(price_data).ledoit_wolf()
                    else:
                        S = risk_models.sample_cov(price_data)
                    
                    # Create views matrix and uncertainty matrix
                    P = np.eye(len(tickers))  # Each view is about one asset
                    Q = np.array([views[ticker] for ticker in tickers])
                    omega = np.diag(np.array(confidences))
                    
                    # Apply Black-Litterman
                    bl = black_litterman.BlackLittermanModel(S, pi=mu_market, P=P, Q=Q, omega=omega)
                    mu_bl = bl.bl_returns()
                    S_bl = bl.bl_cov()
                    
                    mu = mu_bl  # Update expected returns with Black-Litterman

                    # Store final returns for the table
                    st.session_state.bl_table_data['final_returns'] = dict(zip(tickers, mu_bl))
                                        
                    cov_matrix = S_bl
                    
                    st.write("**Final Expected Returns (Black-Litterman adjusted):**")
                    for ticker in tickers:
                        st.write(f"- {ticker}: {mu[ticker]:.2%}")

                else:
                    # Standard approach - just calculate covariance matrix (same as before)
                    st.write("Using your pre-calculated returns with sample covariance...")
                    
                    try:
                        if st.session_state.stabilize_covariance:
                            st.write("Using Ledoit-Wolf shrinkage for covariance matrix...")
                            cov_matrix = risk_models.CovarianceShrinkage(price_data).ledoit_wolf()
                        else:
                            st.write("Using sample covariance matrix...")
                            cov_matrix = risk_models.sample_cov(price_data)
                            
                    except Exception as cov_error:
                        st.warning(f"Standard covariance failed: {cov_error}")
                        st.write("Falling back to Ledoit-Wolf shrinkage...")
                        cov_matrix = risk_models.CovarianceShrinkage(price_data).ledoit_wolf()
                    
                    # Keep your original expected returns (no change needed)
                    st.write("**Expected Returns (unchanged from Asset Analyzer):**")
                    for ticker in tickers:
                        st.write(f"- {ticker}: {mu[ticker]:.2%}")
                
                # Validate results
                if mu.isnull().any():
                    st.error("❌ Some expected returns are NaN. Cannot proceed with optimization.")
                    st.stop()
                
                if (np.diag(cov_matrix) <= 0).any():
                    st.error("❌ Covariance matrix has non-positive diagonal elements. Cannot proceed.")
                    st.stop()
                
                st.success("✅ Expected returns and covariance matrix ready for optimization")
                
            except Exception as e:
                st.error(f"❌ Error calculating covariance matrix: {str(e)}")
                st.write("**Troubleshooting steps:**")
                st.write("1. Check if you have sufficient price history in your selected date range")
                st.write("2. Try enabling 'Stabilize covariance matrix'")
                st.write("3. Consider extending your date range for more data")
                st.stop()
            
            # Apply portfolio constraints
            constraints = []
            for stock in st.session_state.portfolio:
                ticker = stock["ticker"]
                if ticker in tickers:
                    min_weight = stock["min"] / 100  # Convert percentage to decimal
                    max_weight = stock["max"] / 100  # Convert percentage to decimal
                    constraints.append((min_weight, max_weight))
                else:
                    constraints.append((0, 1))  # Default if ticker not found
            
            # Store optimization results
            optimization_results = {}
            
            # Perform optimizations based on selected objectives
            if st.session_state.optimization_objectives["max_sharpe"]:
                try:
                    # Max Sharpe
                    ef_sharpe = EfficientFrontier(mu, cov_matrix, weight_bounds=tuple(constraints))
                    
                    # Add group constraints if enabled
                    if st.session_state.get('enable_asset_groups', False) and st.session_state.get('asset_groups', {}):
                        for group_name, group_data in st.session_state.asset_groups.items():
                            # Find tickers in this group
                            group_tickers = [s["ticker"] for s in st.session_state.portfolio if s.get("group") == group_name]
                            group_indices = [i for i, ticker in enumerate(tickers) if ticker in group_tickers]
                            
                            if group_indices:
                                group_min = group_data["min"] / 100
                                group_max = group_data["max"] / 100
                                ef_sharpe.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) >= group_min)
                                ef_sharpe.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) <= group_max)
                    
                    weights_sharpe = ef_sharpe.max_sharpe()
                    ret_sharpe, vol_sharpe, sharpe_sharpe = ef_sharpe.portfolio_performance()
                    
                    optimization_results["Max Sharpe Ratio"] = {
                        'weights': np.array(list(weights_sharpe.values())),
                        'return': ret_sharpe,
                        'volatility': vol_sharpe,
                        'sharpe': sharpe_sharpe
                    }
                    st.success("✅ Max Sharpe optimization successful!")
                        
                except Exception as e:
                    st.error(f"Error setting up Max Sharpe optimization: {str(e)}")
            
            if st.session_state.optimization_objectives["min_volatility"]:
                try:
                    # Min Volatility  
                    ef_min_vol = EfficientFrontier(mu, cov_matrix, weight_bounds=tuple(constraints))
                    
                    # Add group constraints if enabled
                    if st.session_state.get('enable_asset_groups', False) and st.session_state.get('asset_groups', {}):
                        for group_name, group_data in st.session_state.asset_groups.items():
                            group_tickers = [s["ticker"] for s in st.session_state.portfolio if s.get("group") == group_name]
                            group_indices = [i for i, ticker in enumerate(tickers) if ticker in group_tickers]
                            
                            if group_indices:
                                group_min = group_data["min"] / 100
                                group_max = group_data["max"] / 100
                                ef_min_vol.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) >= group_min)
                                ef_min_vol.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) <= group_max)
                    
                    weights_min_vol = ef_min_vol.min_volatility()
                    ret_min_vol, vol_min_vol, sharpe_min_vol = ef_min_vol.portfolio_performance()
                    
                    optimization_results["Min Volatility"] = {
                        'weights': np.array(list(weights_min_vol.values())),
                        'return': ret_min_vol,
                        'volatility': vol_min_vol,
                        'sharpe': sharpe_min_vol
                    }
                except Exception as e:
                    st.error(f"Error optimizing for Min Volatility: {str(e)}")
        
            if st.session_state.optimization_objectives["max_return_fixed_risk"]:
                try:
                    if st.session_state.get('optimization_type') == "max_sharpe_in_range":
                        # Correct mathematical approach: Boundary analysis without DCP violations
                        vol_min = st.session_state.volatility_min
                        vol_max = st.session_state.volatility_max
                        
                        st.write(f"**Mathematically optimal approach: {vol_min:.1%} to {vol_max:.1%} volatility range**")
                        st.write("Using convex boundary analysis - no DCP violations")
                        
                        try:
                            # Step 1: Find unconstrained max Sharpe portfolio
                            st.write("**Step 1: Computing unconstrained maximum Sharpe ratio...**")
                            
                            ef_unconstrained = EfficientFrontier(mu, cov_matrix, weight_bounds=tuple(constraints))
                            
                            # Add group constraints (these are convex!)
                            if st.session_state.get('enable_asset_groups', False) and st.session_state.get('asset_groups', {}):
                                for group_name, group_data in st.session_state.asset_groups.items():
                                    group_tickers = [s["ticker"] for s in st.session_state.portfolio if s.get("group") == group_name]
                                    group_indices = [i for i, ticker in enumerate(tickers) if ticker in group_tickers]
                                    
                                    if group_indices:
                                        group_min = group_data["min"] / 100
                                        group_max = group_data["max"] / 100
                                        ef_unconstrained.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) >= group_min)
                                        ef_unconstrained.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) <= group_max)
                            
                            # Solve unconstrained max Sharpe (this is convex!)
                            weights_max_sharpe = ef_unconstrained.max_sharpe()
                            ret_unconstrained, vol_unconstrained, sharpe_unconstrained = ef_unconstrained.portfolio_performance()
                            
                            st.write(f"Unconstrained optimum: {sharpe_unconstrained:.4f} Sharpe at {vol_unconstrained:.2%} volatility")
                            
                            # Step 2: Apply boundary logic (mathematically rigorous)
                            if vol_min <= vol_unconstrained <= vol_max:
                                # Case 1: Unconstrained optimum falls within range
                                st.success(f"✅ **Global optimum found!** Unconstrained max Sharpe falls within target range")
                                
                                optimization_results[f"Max Sharpe in {vol_min:.0%}-{vol_max:.0%} Range"] = {
                                    'weights': np.array(list(weights_max_sharpe.values())),
                                    'return': ret_unconstrained,
                                    'volatility': vol_unconstrained,
                                    'sharpe': sharpe_unconstrained
                                }
                                
                            elif vol_unconstrained > vol_max:
                                # Case 2: Unconstrained optimum is too risky
                                st.write(f"⚠️ Unconstrained optimum ({vol_unconstrained:.2%}) exceeds maximum volatility ({vol_max:.2%})")
                                st.write("**Mathematical conclusion**: Optimal solution is at upper boundary")
                                
                                # Solve at upper boundary (convex problem!)
                                ef_upper = EfficientFrontier(mu, cov_matrix, weight_bounds=tuple(constraints))
                                
                                # Add group constraints
                                if st.session_state.get('enable_asset_groups', False) and st.session_state.get('asset_groups', {}):
                                    for group_name, group_data in st.session_state.asset_groups.items():
                                        group_tickers = [s["ticker"] for s in st.session_state.portfolio if s.get("group") == group_name]
                                        group_indices = [i for i, ticker in enumerate(tickers) if ticker in group_tickers]
                                        
                                        if group_indices:
                                            group_min = group_data["min"] / 100
                                            group_max = group_data["max"] / 100
                                            ef_upper.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) >= group_min)
                                            ef_upper.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) <= group_max)
                                
                                weights_upper = ef_upper.efficient_risk(vol_max)
                                ret_upper, vol_upper, sharpe_upper = ef_upper.portfolio_performance()
                                
                                optimization_results[f"Max Sharpe in {vol_min:.0%}-{vol_max:.0%} Range"] = {
                                    'weights': np.array(list(weights_upper.values())),
                                    'return': ret_upper,
                                    'volatility': vol_upper,
                                    'sharpe': sharpe_upper
                                }
                                
                            else:  # vol_unconstrained < vol_min
                                # Case 3: Unconstrained optimum is too conservative
                                st.write(f"⚠️ Unconstrained optimum ({vol_unconstrained:.2%}) below minimum volatility ({vol_min:.2%})")
                                st.write("**Mathematical conclusion**: Optimal solution is at lower boundary")
                                
                                # Solve at lower boundary (convex problem!)
                                ef_lower = EfficientFrontier(mu, cov_matrix, weight_bounds=tuple(constraints))
                                
                                # Add group constraints
                                if st.session_state.get('enable_asset_groups', False) and st.session_state.get('asset_groups', {}):
                                    for group_name, group_data in st.session_state.asset_groups.items():
                                        group_tickers = [s["ticker"] for s in st.session_state.portfolio if s.get("group") == group_name]
                                        group_indices = [i for i, ticker in enumerate(tickers) if ticker in group_tickers]
                                        
                                        if group_indices:
                                            group_min = group_data["min"] / 100
                                            group_max = group_data["max"] / 100
                                            ef_lower.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) >= group_min)
                                            ef_lower.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) <= group_max)
                                
                                weights_lower = ef_lower.efficient_risk(vol_min)
                                ret_lower, vol_lower, sharpe_lower = ef_lower.portfolio_performance()
                                
                                optimization_results[f"Max Sharpe in {vol_min:.0%}-{vol_max:.0%} Range"] = {
                                    'weights': np.array(list(weights_lower.values())),
                                    'return': ret_lower,
                                    'volatility': vol_lower,
                                    'sharpe': sharpe_lower
                                }
                            
                        except Exception as e:
                            st.error(f"❌ Convex optimization failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                
                    else:
                        # Standard: Maximize return at fixed risk
                        ef_max_ret = EfficientFrontier(mu, cov_matrix, weight_bounds=tuple(constraints))
                        
                        # Add group constraints if enabled
                        if st.session_state.get('enable_asset_groups', False) and st.session_state.get('asset_groups', {}):
                            for group_name, group_data in st.session_state.asset_groups.items():
                                group_tickers = [s["ticker"] for s in st.session_state.portfolio if s.get("group") == group_name]
                                group_indices = [i for i, ticker in enumerate(tickers) if ticker in group_tickers]
                                
                                if group_indices:
                                    group_min = group_data["min"] / 100
                                    group_max = group_data["max"] / 100
                                    ef_max_ret.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) >= group_min)
                                    ef_max_ret.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) <= group_max)
                        
                        weights_max_ret = ef_max_ret.efficient_risk(st.session_state.target_risk)
                        ret_max_ret, vol_max_ret, sharpe_max_ret = ef_max_ret.portfolio_performance()
                        
                        optimization_results[f"Max Return @ {st.session_state.target_risk:.1%} Risk"] = {
                            'weights': np.array(list(weights_max_ret.values())),
                            'return': ret_max_ret,
                            'volatility': vol_max_ret,
                            'sharpe': sharpe_max_ret
                        }
                        
                except Exception as e:
                    st.error(f"Error optimizing for Max Return @ Fixed Risk: {str(e)}")

            if st.session_state.optimization_objectives["min_risk_fixed_return"]:
                try:
                    # Min Risk @ Fixed Return
                    ef_min_risk = EfficientFrontier(mu, cov_matrix, weight_bounds=tuple(constraints))
                    
                    # Add group constraints if enabled
                    if st.session_state.get('enable_asset_groups', False) and st.session_state.get('asset_groups', {}):
                        for group_name, group_data in st.session_state.asset_groups.items():
                            group_tickers = [s["ticker"] for s in st.session_state.portfolio if s.get("group") == group_name]
                            group_indices = [i for i, ticker in enumerate(tickers) if ticker in group_tickers]
                            
                            if group_indices:
                                group_min = group_data["min"] / 100
                                group_max = group_data["max"] / 100
                                ef_min_risk.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) >= group_min)
                                ef_min_risk.add_constraint(lambda w, indices=group_indices: sum(w[i] for i in indices) <= group_max)
                    
                    weights_min_risk = ef_min_risk.efficient_return(st.session_state.target_return)
                    ret_min_risk, vol_min_risk, sharpe_min_risk = ef_min_risk.portfolio_performance()

                    # CHECK: If actual return is significantly higher than target, warn user
                    target_return_pct = st.session_state.target_return * 100
                    actual_return_pct = ret_min_risk * 100
                    
                    if actual_return_pct > target_return_pct + 1:  # More than 1% difference
                        warning_message = f"Note: The minimum-risk portfolio that meets your {target_return_pct:.1f}% return target actually has a return of {actual_return_pct:.1f}%. This means no efficient portfolio exists with exactly {target_return_pct:.1f}% return - the feasible region starts higher."
                        st.warning(warning_message)
                        
                        # Store the warning for later display
                        st.session_state.min_risk_warning = warning_message
                    else:
                        # Clear any previous warning
                        st.session_state.min_risk_warning = None

                    optimization_results[f"Min Risk @ {st.session_state.target_return:.1%} Return"] = {
                        'weights': np.array(list(weights_min_risk.values())),
                        'return': ret_min_risk,
                        'volatility': vol_min_risk,
                        'sharpe': sharpe_min_risk
                    }
                except Exception as e:
                    st.error(f"Error optimizing for Min Risk @ Fixed Return: {str(e)}")
                    optimization_results[f"Min Risk @ {st.session_state.target_return:.1%} Return"] = {
                        'weights': np.array(list(weights_min_risk.values())),
                        'return': ret_min_risk,
                        'volatility': vol_min_risk,
                        'sharpe': sharpe_min_risk
                    }
                except Exception as e:
                    st.error(f"Error optimizing for Min Risk @ Fixed Return: {str(e)}")

            # ADD THIS NEW CODE HERE:
            # Store variables for use outside the status block
            if optimization_results:
                st.session_state.temp_optimization_results = optimization_results
                st.session_state.temp_mu = mu
                st.session_state.temp_cov_matrix = cov_matrix
                st.session_state.temp_tickers = tickers

            # Mark status as complete and collapse
            status.update(label="✅ Portfolio optimization complete!", state="complete", expanded=False)         

            
        except Exception as e:
            status.update(label="❌ Optimization failed!", state="error", expanded=False)
            st.error(f"❌ Optimization failed: {str(e)}")
            st.write("Please check your portfolio constraints and try again.")

# Display results OUTSIDE the status block (so they stay visible)
    if hasattr(st.session_state, 'temp_optimization_results') and st.session_state.temp_optimization_results:
        optimization_results = st.session_state.temp_optimization_results
        mu = st.session_state.temp_mu
        cov_matrix = st.session_state.temp_cov_matrix
        tickers = st.session_state.temp_tickers
        
        # Try to create efficient frontier plot
        try:
            ef_plot = EfficientFrontier(mu, cov_matrix)
            weights_for_plot = {name: result['weights'] for name, result in optimization_results.items()}
            
            fig = create_efficient_frontier_plot(ef_plot, tickers, weights_for_plot)
            st.plotly_chart(fig, use_container_width=True, key="efficient_frontier_plot")

        except Exception as e:
            if st.session_state.use_black_litterman and st.session_state.get('bl_use_market_implied', False):
                delta_value = st.session_state.get('bl_delta', 2.5)
                st.warning(f"Efficient frontier plot unavailable with δ={delta_value} (solver limitations)")
                st.info("Try reducing Risk Aversion Coefficient to δ=1.0-1.5 for visualization.")
            else:
                st.warning("Could not generate efficient frontier plot due to numerical issues.")


        st.subheader("Optimization Results")
        
        for strategy_name, result in optimization_results.items():
            st.write("---")
            # Strategy title FIRST
            st.subheader(strategy_name)
            
            # ADD WARNING HERE if it's the Min Risk strategy
            if "Min Risk @" in strategy_name and hasattr(st.session_state, 'min_risk_warning') and st.session_state.min_risk_warning:
                st.warning(st.session_state.min_risk_warning)

            # Performance metrics SECOND
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Annual Return", f"{result['return']:.2%}")
            with col2:
                st.metric("Annual Volatility", f"{result['volatility']:.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{result['sharpe']:.3f}")
            
            # Portfolio allocation (but don't repeat the title)
            portfolio_groups = st.session_state.get('asset_groups', {}) if st.session_state.get('enable_asset_groups', False) else None
            display_portfolio_allocation(result['weights'], tickers, strategy_name, portfolio_groups) 

        # Black-Litterman detailed breakdown table
        if st.session_state.use_black_litterman and hasattr(st.session_state, 'bl_table_data'):
            with st.expander("📊 View Black-Litterman Model Details", expanded=False):
                st.write("### Black-Litterman Process Breakdown")
                
                # Get the stored data
                bl_data = st.session_state.bl_table_data
                
                # Build the table data
                table_data = []
                
                for ticker in bl_data['tickers']:
                    row = {'Ticker': ticker}
                    
                    # Market Cap (if applicable)
                    if bl_data['prior_type'] == 'market_implied':
                        market_cap_info = bl_data['market_caps'].get(ticker, 'N/A')
                        if isinstance(market_cap_info, (int, float)) and market_cap_info > 0:
                            row['Market Cap'] = f"${market_cap_info:,.0f}"
                        else:
                            row['Market Cap'] = str(market_cap_info)
                        
                        # Market Cap Weight
                        weight = bl_data['market_weights'].get(ticker, 0)
                        row['Market Cap Weight'] = f"{weight:.2%}"
                        
                        # Market-Implied Return
                        market_return = bl_data['market_implied_returns'].get(ticker, 0)
                        row['Market-Implied Return'] = f"{market_return:.2%}"
                        
                        # Historical Return (N/A for market-implied)
                        row['Historical Return'] = 'N/A'
                        
                    else:  # historical
                        row['Market Cap'] = 'N/A'
                        row['Market Cap Weight'] = 'N/A'
                        row['Market-Implied Return'] = 'N/A'
                        
                        # Historical Return
                        hist_return = bl_data['historical_returns'].get(ticker, 0)
                        row['Historical Return'] = f"{hist_return:.2%}"
                    
                    # User Views (always applicable)
                    user_view = bl_data['user_views'].get(ticker, {})
                    row['Your View'] = f"{user_view.get('return', 0):.1f}%"
                    row['Your Confidence'] = f"{user_view.get('confidence', 0)}%"
                    
                    # Final BL-Adjusted Return
                    final_return = bl_data['final_returns'].get(ticker, 0) if bl_data['final_returns'] else 0
                    row['Final BL-Adjusted Return'] = f"{final_return:.2%}"
                    
                    table_data.append(row)
                
                # Create DataFrame
                bl_details_df = pd.DataFrame(table_data)
                
                # Determine which columns to show based on prior type
                if bl_data['prior_type'] == 'market_implied':
                    columns_to_show = [
                        'Ticker', 'Market Cap', 'Market Cap Weight', 'Market-Implied Return',
                        'Your View', 'Your Confidence', 'Final BL-Adjusted Return'
                    ]
                else:  # historical
                    columns_to_show = [
                        'Ticker', 'Historical Return', 'Your View', 'Your Confidence', 'Final BL-Adjusted Return'
                    ]
                
                # Display only relevant columns
                display_df = bl_details_df[columns_to_show]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Add some explanatory text
                if bl_data['prior_type'] == 'market_implied':
                    st.info("💡 **Market-Implied Returns** are calculated using the formula: Π = δ·Σ·w_mkt, where δ is the risk aversion coefficient, Σ is the covariance matrix, and w_mkt are the market cap weights.")
                else:
                    st.info("💡 **Historical Returns** from your Asset Analyzer are used as the baseline prior, then adjusted based on your market views and confidence levels.")

        # === Correlation Matrix (Plotly-style like Page 4) + Downloads ===
        with st.expander("📊 View Correlation Matrix & Downloads", expanded=False):
            import numpy as np
            import plotly.express as px
            import plotly.io as pio

            st.write("### Correlation Matrix")

            # Inform user which covariance powered the optimizer
            if st.session_state.stabilize_covariance:
                st.info("💡 **Ledoit–Wolf Shrinkage** covariance was used for optimization. The correlation below is derived from that matrix.")
            else:
                st.info("💡 **Sample Covariance** was used for optimization. The correlation below is derived from that matrix.")

            # Build DataFrames from the exact matrix used upstream on Page 3
            cov_df = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)

            # Safe cov→corr conversion (handles zero-vol assets)
            std = np.sqrt(np.diag(cov_matrix))
            std = np.where(std == 0, np.nan, std)
            corr_matrix = cov_matrix / np.outer(std, std)
            corr_df = pd.DataFrame(corr_matrix, index=tickers, columns=tickers)

            # ---- Plotly heatmap (match Page 4 look: RdBu + visible colorbar) ----
            # Clamp to [-1, 1] for stability in display (NaNs remain NaN for styling)
            corr_plot = corr_df.copy()
            corr_plot = corr_plot.clip(lower=-1.0, upper=1.0)

            fig = px.imshow(
                corr_plot,
                text_auto=True,                 # show values in cells (like Page 4)
                aspect="auto",
                color_continuous_scale="RdBu",  # same palette as Page 4
                zmin=-1, zmax=1,                # fixed range with centered white @ 0
                labels=dict(color="Correlation")
            )

            # Improve readability: smaller font inside cells, rotate x labels
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
            st.caption("-1 = move opposite · 0 = weak/none · +1 = move together")

            st.write("---")
            st.write("### Downloads")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇️ Download **Covariance Matrix** (CSV)",
                    data=cov_df.to_csv(index=True).encode("utf-8"),
                    file_name="covariance_matrix.csv",
                    mime="text/csv",
                    help="Raw matrix actually used by the optimizer"
                )
            with c2:
                st.download_button(
                    "⬇️ Download **Correlation Matrix** (CSV)",
                    data=corr_df.to_csv(index=True).encode("utf-8"),
                    file_name="correlation_matrix.csv",
                    mime="text/csv"
                )

            st.write("**Color Guide:**")
            st.write("- 🔴 Negative correlation (diversifies risk)")
            st.write("- ⚪ Near zero (weak linear relationship)")
            st.write("- 🔵 Positive correlation (move together)")

    
    else:
        st.error("❌ No optimization results generated. Please check your constraints and try again.")

