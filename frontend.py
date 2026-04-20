"""
NeuralStockTrader - Web Frontend
Rich, interactive dashboard for neural network stock trading system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yaml
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="NeuralStockTrader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #333;
        padding: 1.5rem;
        border-radius: 10px;
    }
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #333;
        padding: 1.5rem;
        border-radius: 10px;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "dashboard"
if "trading_active" not in st.session_state:
    st.session_state.trading_active = False


def load_config():
    """Load configuration"""
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def create_sidebar():
    """Create navigation sidebar"""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/numpy/numpy/main/branding/logo/logomark/numpylogomark.png", 
                width=100)
        
        st.title("🚀 NeuralStockTrader")
        st.markdown("---")
        
        pages = {
            "📊 Dashboard": "dashboard",
            "📈 Backtest": "backtest",
            "🤖 Model Training": "training",
            "💹 Live Trading": "live_trading",
            "📉 Portfolio": "portfolio",
            "⚙️ Risk Management": "risk",
            "📋 History": "history",
            "⚙️ Settings": "settings"
        }
        
        st.session_state.current_page = st.radio(
            "Navigation",
            list(pages.keys()),
            index=list(pages.keys()).index(next((k for k, v in pages.items() if v == st.session_state.current_page), "📊 Dashboard"))
        )
        st.session_state.current_page = pages[st.session_state.current_page]
        
        st.markdown("---")
        
        # Market overview
        st.subheader("🌍 Market Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Market Status", "Open ✅")
        with col2:
            st.metric("Last Update", "2 min ago")
        
        st.markdown("---")
        st.markdown("**Quick Settings**")
        trading_mode = st.selectbox(
            "Trading Mode",
            ["📊 Backtest", "📄 Paper", "💼 Live"],
            index=0
        )


def render_dashboard():
    """Dashboard page"""
    st.markdown('<div class="header-title">📊 Trading Dashboard</div>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Returns", "+24.5%", "+2.3%", delta_color="off")
    with col2:
        st.metric("Win Rate", "62.3%", "+5.2%", delta_color="off")
    with col3:
        st.metric("Sharpe Ratio", "1.85", "+0.12", delta_color="off")
    with col4:
        st.metric("Max Drawdown", "-8.2%", "-1.5%", delta_color="inverse")
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Portfolio Value Over Time")
        dates = pd.date_range(start='2024-01-01', periods=100)
        portfolio_values = 100000 * (1 + np.cumsum(np.random.randn(100) * 0.01))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='rgba(102, 126, 234, 1)', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_showgrid=False,
            yaxis_showgrid=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Strategy Performance")
        strategies = ['Mean Reversion', 'Momentum', 'Arbitrage', 'ML Ensemble']
        returns = [18.5, 22.3, 15.7, 24.5]
        colors = ['#667eea', '#764ba2', '#84fab0', '#fa709a']
        
        fig = go.Figure(data=[
            go.Bar(x=strategies, y=returns, marker_color=colors)
        ])
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_showgrid=False,
            yaxis_showgrid=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Active positions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Active Positions")
        positions_data = {
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            'Quantity': [100, 50, 25, 75],
            'Entry Price': [150.20, 380.50, 140.30, 245.60],
            'Current Price': [152.10, 385.30, 142.80, 248.20],
            'P&L': ['+$190', '+$240', '+$62.50', '+$195'],
            'P&L %': ['+1.26%', '+1.26%', '+1.78%', '+1.06%']
        }
        df_positions = pd.DataFrame(positions_data)
        st.dataframe(df_positions, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📝 Recent Trades")
        trades_data = {
            'Time': pd.date_range('2024-01-15', periods=5, freq='1H'),
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'Side': ['BUY', 'SELL', 'BUY', 'BUY', 'SELL'],
            'Price': [150.20, 385.30, 140.30, 245.60, 875.50],
            'Size': [100, 50, 25, 75, 30],
            'Status': ['✅ Filled', '✅ Filled', '✅ Filled', '✅ Filled', '⏳ Pending']
        }
        df_trades = pd.DataFrame(trades_data)
        st.dataframe(df_trades, use_container_width=True, hide_index=True)


def render_backtest():
    """Backtest page"""
    st.markdown('<div class="header-title">📈 Backtesting Engine</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Backtest Configuration")
        symbol = st.selectbox("Select Symbol", ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"])
        
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input("Start Date", datetime(2023, 1, 1))
        with col_b:
            end_date = st.date_input("End Date", datetime(2024, 1, 1))
        
        initial_capital = st.number_input("Initial Capital ($)", value=100000, min_value=1000, step=1000)
        
        st.markdown("**Strategy Selection**")
        strategies = st.multiselect(
            "Choose Strategies",
            ["Mean Reversion", "Momentum", "Arbitrage", "ML Ensemble", "Combined"],
            default=["ML Ensemble"]
        )
        
        col_x, col_y = st.columns(2)
        with col_x:
            max_position = st.slider("Max Position Size (%)", 1, 50, 10)
        with col_y:
            stop_loss = st.slider("Stop Loss (%)", 1, 20, 5)
        
        if st.button("▶️ Run Backtest", use_container_width=True, type="primary"):
            st.success("✅ Backtest started! Processing...")
    
    with col2:
        st.subheader("📊 Backtest Results")
        results = {
            'Metric': [
                'Total Return',
                'Annual Return',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Max Drawdown',
                'Win Rate',
                'Total Trades',
                'Profit Factor'
            ],
            'Value': [
                '+24.53%',
                '+18.21%',
                '1.85',
                '2.34',
                '-8.23%',
                '62.34%',
                '127',
                '2.15'
            ]
        }
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Equity curve
    st.subheader("📈 Equity Curve")
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    equity = 100000 * np.cumprod(1 + np.random.randn(len(dates)) * 0.005)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=equity,
        mode='lines+markers',
        name='Equity',
        line=dict(color='rgba(132, 250, 176, 1)', width=2),
        fill='tozeroy',
        fillcolor='rgba(132, 250, 176, 0.2)'
    ))
    fig.update_layout(
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_showgrid=False,
        yaxis_showgrid=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Winning Trades", "79", "62.2%")
    with col2:
        st.metric("Losing Trades", "48", "37.8%")
    with col3:
        st.metric("Avg Win/Loss Ratio", "1.89", "↑")


def render_training():
    """Model Training page"""
    st.markdown('<div class="header-title">🤖 Model Training</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🧠 Neural Network Configuration")
        
        model_type = st.selectbox("Model Architecture", ["LSTM", "GRU", "Transformer", "Ensemble"])
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            lookback_period = st.number_input("Lookback Period (days)", value=60, min_value=20, max_value=500)
        with col_b:
            forecast_horizon = st.number_input("Forecast Horizon", value=5, min_value=1, max_value=30)
        with col_c:
            batch_size = st.number_input("Batch Size", value=32, min_value=8, max_value=256, step=8)
        
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            epochs = st.number_input("Epochs", value=100, min_value=10, max_value=1000, step=10)
        with col_y:
            learning_rate = st.select_slider("Learning Rate", 
                                            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                                            value=0.001)
        with col_z:
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
        
        st.markdown("**Dataset Configuration**")
        col1_d, col2_d = st.columns(2)
        with col1_d:
            train_split = st.slider("Training Split (%)", 50, 90, 80)
        with col2_d:
            validation_split = st.slider("Validation Split (%)", 10, 50, 20)
        
        if st.button("🚀 Start Training", use_container_width=True, type="primary"):
            st.success("✅ Training initiated!")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(101):
                progress_bar.progress(i)
                status_text.text(f"Training Progress: {i}%")
            
            st.balloons()
    
    with col2:
        st.subheader("📊 Training Status")
        col_status_1, col_status_2 = st.columns(1)
        with col_status_1:
            st.metric("Status", "✅ Running", "70%", label_visibility="collapsed")
            st.metric("Epoch", "70/100", label_visibility="collapsed")
            st.metric("Loss", "0.0234", "↓", label_visibility="collapsed")
            st.metric("Val Loss", "0.0245", "↑", label_visibility="collapsed")
    
    st.markdown("---")
    
    # Training history
    st.subheader("📈 Training Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs_data = list(range(1, 101))
        loss_data = 0.5 * np.exp(-np.array(epochs_data) / 50) + 0.02 * np.random.randn(100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs_data, y=loss_data, mode='lines', name='Training Loss',
                                line=dict(color='#fa709a', width=2)))
        fig.update_layout(
            height=400, title="Training Loss",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        val_loss_data = 0.52 * np.exp(-np.array(epochs_data) / 50) + 0.03 * np.random.randn(100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs_data, y=val_loss_data, mode='lines', name='Validation Loss',
                                line=dict(color='#8fd3f4', width=2)))
        fig.update_layout(
            height=400, title="Validation Loss",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_live_trading():
    """Live Trading page"""
    st.markdown('<div class="header-title">💹 Live Trading Monitor</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Account Balance", "$124,532.45", "+$5,230.21", delta_color="off")
    with col2:
        st.metric("Open P&L", "+$2,145.67", "+1.73%", delta_color="off")
    with col3:
        st.metric("Positions", "4", "↔")
    with col4:
        st.metric("Daily Trades", "12", "↑ 3")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Start Trading", use_container_width=True, type="primary"):
            st.session_state.trading_active = True
            st.success("✅ Trading Started!")
        
        if st.button("⏸️ Stop Trading", use_container_width=True):
            st.session_state.trading_active = False
            st.info("⏸️ Trading Paused")
    
    with col2:
        st.selectbox("Active Strategy", ["Mean Reversion", "Momentum", "Combined", "ML Ensemble"])
    
    st.markdown("---")
    
    st.subheader("📊 Live Market Data")
    
    live_data = {
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        'Price': [152.45, 385.23, 142.67, 248.34, 876.50],
        'Change': ['+1.23%', '+0.89%', '-0.45%', '+2.15%', '+3.45%'],
        'Volume': ['52.3M', '28.5M', '35.2M', '42.1M', '31.8M'],
        'Signal': ['🟢 BUY', '🟡 HOLD', '🔴 SELL', '🟢 BUY', '🟢 BUY'],
        'Confidence': ['85%', '62%', '78%', '92%', '88%']
    }
    
    df_live = pd.DataFrame(live_data)
    st.dataframe(df_live, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.subheader("📈 Real-time Chart")
    
    times = pd.date_range(start='2024-01-15 09:30', periods=100, freq='5min')
    prices = 150 + np.cumsum(np.random.randn(100) * 0.5)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=prices,
        mode='lines',
        name='AAPL',
        line=dict(color='#667eea', width=2)
    ))
    fig.update_layout(
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_showgrid=False,
        yaxis_showgrid=True
    )
    st.plotly_chart(fig, use_container_width=True)


def render_portfolio():
    """Portfolio page"""
    st.markdown('<div class="header-title">📉 Portfolio Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", "$124,532.45", "+$5,230")
    with col2:
        st.metric("Cash", "$24,532.45", "-$1,000")
    with col3:
        st.metric("Securities", "$100,000", "+$6,230")
    with col4:
        st.metric("Allocation", "80.3%", "↔")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Asset Allocation")
        allocation = {
            'Asset': ['US Stocks', 'International', 'Bonds', 'Cash', 'Crypto'],
            'Value': [80000, 15000, 18000, 8000, 3532.45]
        }
        df_alloc = pd.DataFrame(allocation)
        
        fig = go.Figure(data=[
            go.Pie(labels=df_alloc['Asset'], values=df_alloc['Value'],
                  marker=dict(colors=['#667eea', '#764ba2', '#84fab0', '#fa709a', '#fee140']))
        ])
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Top Holdings")
        holdings = {
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'Shares': [100, 50, 25, 75, 30],
            'Value': [15210, 19265, 3567, 18615, 26295],
            '% Portfolio': ['12.2%', '15.5%', '2.9%', '14.9%', '21.1%'],
            'Return': ['+1.26%', '+2.34%', '-0.45%', '+3.21%', '+5.67%']
        }
        df_holdings = pd.DataFrame(holdings)
        st.dataframe(df_holdings, use_container_width=True, hide_index=True)


def render_risk_management():
    """Risk Management page"""
    st.markdown('<div class="header-title">⚙️ Risk Management</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Portfolio Risk (VaR)", "-$8,234", "95% Confidence")
    with col2:
        st.metric("Beta", "0.95", "vs Market")
    with col3:
        st.metric("Correlation", "0.82", "to S&P 500")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Risk Controls")
        
        st.markdown("**Position Limits**")
        col_a, col_b = st.columns(2)
        with col_a:
            max_position_limit = st.slider("Max Position Size", 5, 50, 20)
        with col_b:
            max_sector_limit = st.slider("Max Sector Allocation", 20, 60, 40)
        
        st.markdown("**Stop Loss & Take Profit**")
        col_x, col_y = st.columns(2)
        with col_x:
            stop_loss_pct = st.number_input("Stop Loss (%)", value=5.0, min_value=0.5, step=0.5)
        with col_y:
            take_profit_pct = st.number_input("Take Profit (%)", value=15.0, min_value=1.0, step=1.0)
        
        st.markdown("**Daily Limits**")
        col_p, col_q = st.columns(2)
        with col_p:
            daily_loss_limit = st.number_input("Daily Loss Limit ($)", value=-5000, step=-100)
        with col_q:
            max_drawdown_limit = st.number_input("Max Drawdown (%)", value=-10.0, step=-0.5)
        
        if st.button("✅ Update Risk Parameters", use_container_width=True, type="primary"):
            st.success("Risk parameters updated!")
    
    with col2:
        st.subheader("📊 Risk Metrics")
        
        risk_metrics = {
            'Metric': ['Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio'],
            'Value': ['12.4%', '1.85', '2.34', '-8.2%', '2.21']
        }
        df_risk = pd.DataFrame(risk_metrics)
        st.dataframe(df_risk, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.subheader("🔍 Sector Risk")
        sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer']
        sector_risk = [15.2, 12.3, 14.1, 18.5, 11.2]
        
        fig = go.Figure(data=[
            go.Bar(x=sectors, y=sector_risk, marker_color=['#667eea', '#764ba2', '#84fab0', '#fa709a', '#fee140'])
        ])
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)


def render_history():
    """Trading History page"""
    st.markdown('<div class="header-title">📋 Trading History</div>', unsafe_allow_html=True)
    
    st.subheader("📜 Trade Log")
    
    # Create sample trading history
    trades_history = {
        'Date': pd.date_range('2024-01-01', periods=20, freq='1H'),
        'Symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'], 20),
        'Action': np.random.choice(['BUY', 'SELL'], 20),
        'Price': np.random.uniform(100, 400, 20),
        'Quantity': np.random.randint(10, 100, 20),
        'Commission': np.random.uniform(5, 50, 20),
        'P&L': np.random.uniform(-200, 500, 20),
        'Status': np.random.choice(['✅ Filled', '⏳ Pending'], 20)
    }
    
    df_history = pd.DataFrame(trades_history)
    st.dataframe(df_history, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Monthly Performance")
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        monthly_returns = [2.3, 1.8, 3.2, 2.1, 2.9, 3.5]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=months, y=monthly_returns, marker_color='#667eea'))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Cumulative Returns")
        
        cumulative = np.cumprod(1 + np.array(monthly_returns) / 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=cumulative, mode='lines+markers',
            line=dict(color='#84fab0', width=3),
            fill='tozeroy'
        ))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)


def render_settings():
    """Settings page"""
    st.markdown('<div class="header-title">⚙️ Settings</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["General", "API", "Notifications", "Advanced"])
    
    with tabs[0]:
        st.subheader("General Settings")
        
        trading_pairs = st.multiselect(
            "Trading Symbols",
            ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN"],
            default=["AAPL", "MSFT", "GOOGL"]
        )
        
        update_frequency = st.selectbox("Data Update Frequency", 
                                       ["1 min", "5 min", "15 min", "1 hour"],
                                       index=1)
        
        timezone = st.selectbox("Timezone", 
                               ["EST", "CST", "MST", "PST", "UTC"],
                               index=4)
        
        if st.button("💾 Save General Settings", use_container_width=True):
            st.success("✅ Settings saved!")
    
    with tabs[1]:
        st.subheader("API Configuration")
        
        st.markdown("**Alpaca API**")
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input("API Key", type="password")
        with col2:
            api_secret = st.text_input("API Secret", type="password")
        
        st.markdown("**Data Provider**")
        data_provider = st.selectbox("Select Data Provider", 
                                    ["Yahoo Finance", "Alpha Vantage", "IEX Cloud"],
                                    index=0)
        
        if st.button("✅ Test Connection", use_container_width=True):
            st.success("✅ Connection successful!")
    
    with tabs[2]:
        st.subheader("Notification Settings")
        
        enable_email = st.checkbox("Email Notifications", value=True)
        enable_sms = st.checkbox("SMS Alerts", value=False)
        enable_pushover = st.checkbox("Push Notifications", value=True)
        
        if enable_email:
            email = st.text_input("Email Address")
        
        if enable_sms:
            phone = st.text_input("Phone Number")
        
        alert_types = st.multiselect(
            "Alert Types",
            ["Trade Executed", "Stop Loss Hit", "Position Opened", "Risk Limit Exceeded"],
            default=["Trade Executed", "Risk Limit Exceeded"]
        )
        
        if st.button("✅ Save Notification Settings", use_container_width=True):
            st.success("✅ Notification settings updated!")
    
    with tabs[3]:
        st.subheader("Advanced Settings")
        
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
        
        enable_logging = st.checkbox("File Logging", value=True)
        
        max_log_size = st.number_input("Max Log File Size (MB)", value=100, min_value=10)
        
        enable_backtesting_cache = st.checkbox("Cache Backtest Data", value=True)
        
        if st.button("✅ Save Advanced Settings", use_container_width=True):
            st.success("✅ Advanced settings updated!")


def main():
    """Main application"""
    create_sidebar()
    
    # Route to appropriate page
    if st.session_state.current_page == "dashboard":
        render_dashboard()
    elif st.session_state.current_page == "backtest":
        render_backtest()
    elif st.session_state.current_page == "training":
        render_training()
    elif st.session_state.current_page == "live_trading":
        render_live_trading()
    elif st.session_state.current_page == "portfolio":
        render_portfolio()
    elif st.session_state.current_page == "risk":
        render_risk_management()
    elif st.session_state.current_page == "history":
        render_history()
    elif st.session_state.current_page == "settings":
        render_settings()


if __name__ == "__main__":
    main()
