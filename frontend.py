"""
NeuralStockTrader - redesigned Streamlit frontend.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml


st.set_page_config(
    page_title="NeuralStockTrader",
    page_icon="NS",
    layout="wide",
    initial_sidebar_state="expanded",
)


PALETTE = {
    "bg": "#07111f",
    "panel": "#0f1b2d",
    "panel_alt": "#13233b",
    "stroke": "#223758",
    "text": "#f4f7fb",
    "muted": "#90a5c4",
    "accent": "#5ae4a8",
    "accent_alt": "#6bc7ff",
    "warning": "#ffb454",
    "danger": "#ff6b7a",
    "purple": "#7b8cff",
}


if "current_page" not in st.session_state:
    st.session_state.current_page = "overview"
if "trading_active" not in st.session_state:
    st.session_state.trading_active = False


def load_config():
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    return {}


def build_market_data(symbols):
    rng = np.random.default_rng(7)
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end, periods=180, freq="D")

    base_prices = {
        "AAPL": 192.4,
        "MSFT": 421.2,
        "GOOGL": 168.7,
        "TSLA": 181.9,
        "NVDA": 911.5,
        "META": 498.1,
        "AMZN": 186.8,
    }

    market = {}
    for idx, symbol in enumerate(symbols):
        drift = 0.0007 + idx * 0.00005
        shock = rng.normal(drift, 0.012, len(dates))
        series = base_prices.get(symbol, 150.0) * np.cumprod(1 + shock)
        volume = rng.integers(8_000_000, 45_000_000, len(dates))
        signal_strength = np.clip(50 + np.cumsum(rng.normal(0, 1.5, len(dates))), 12, 92)
        market[symbol] = pd.DataFrame(
            {
                "date": dates,
                "close": series,
                "volume": volume,
                "signal_strength": signal_strength,
            }
        )
    return market


def build_app_state(config):
    symbols = config.get("data", {}).get("symbols", ["AAPL", "MSFT", "GOOGL", "TSLA"])
    watchlist = list(dict.fromkeys(symbols + ["NVDA", "META", "AMZN"]))[:7]
    market = build_market_data(watchlist)

    primary = market[watchlist[0]].copy()
    benchmark = market[watchlist[1]].copy()
    equity_curve = 100000 * np.cumprod(1 + np.random.default_rng(11).normal(0.0011, 0.006, len(primary)))
    benchmark_curve = 100000 * np.cumprod(1 + np.random.default_rng(21).normal(0.0006, 0.005, len(primary)))
    exposure_curve = np.clip(52 + np.sin(np.linspace(0, 9, len(primary))) * 18, 24, 91)

    positions = pd.DataFrame(
        [
            ["NVDA", "AI Compute", 28, 911.50, 948.20, 2.78, "High momentum"],
            ["MSFT", "Platform", 42, 421.20, 438.10, 3.31, "Earnings drift"],
            ["AAPL", "Consumer", 65, 192.40, 198.65, 2.11, "Stable carry"],
            ["AMZN", "Retail Cloud", 34, 186.80, 190.92, 1.42, "Volume expansion"],
        ],
        columns=["Symbol", "Theme", "Qty", "Entry", "Last", "Edge", "Narrative"],
    )
    positions["Market Value"] = positions["Qty"] * positions["Last"]
    positions["PnL"] = (positions["Last"] - positions["Entry"]) * positions["Qty"]
    positions["PnL %"] = ((positions["Last"] / positions["Entry"]) - 1) * 100

    trades = pd.DataFrame(
        [
            ["09:45", "NVDA", "Buy", "Model breakout", 26, 944.20, "Filled"],
            ["10:12", "META", "Trim", "Crowding risk", 14, 503.90, "Filled"],
            ["11:05", "TSLA", "Buy", "Volatility capture", 18, 184.70, "Pending"],
            ["12:40", "MSFT", "Add", "Quality momentum", 12, 437.50, "Filled"],
            ["13:18", "AAPL", "Hedge", "Reduce beta", 20, 197.85, "Filled"],
        ],
        columns=["Time", "Symbol", "Action", "Reason", "Qty", "Price", "Status"],
    )

    agents = pd.DataFrame(
        [
            ["Neural Core", "Monitoring regime shift", "Stable", 0.87],
            ["Execution Router", "Optimizing order split", "Alert", 0.74],
            ["Risk Sentinel", "Sizing down cyclicals", "Stable", 0.92],
            ["Macro Lens", "Tracking rates narrative", "Watch", 0.61],
        ],
        columns=["Module", "Focus", "State", "Confidence"],
    )

    monthly_returns = pd.DataFrame(
        {
            "Month": ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"],
            "Return": [1.8, 2.4, 3.2, 1.4, 4.1, 2.7],
        }
    )

    return {
        "watchlist": watchlist,
        "market": market,
        "equity_dates": primary["date"],
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "exposure_curve": exposure_curve,
        "positions": positions,
        "trades": trades,
        "agents": agents,
        "monthly_returns": monthly_returns,
    }


def inject_styles():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {{
            --bg: {PALETTE["bg"]};
            --panel: {PALETTE["panel"]};
            --panel-alt: {PALETTE["panel_alt"]};
            --stroke: {PALETTE["stroke"]};
            --text: {PALETTE["text"]};
            --muted: {PALETTE["muted"]};
            --accent: {PALETTE["accent"]};
            --accent-alt: {PALETTE["accent_alt"]};
            --warning: {PALETTE["warning"]};
            --danger: {PALETTE["danger"]};
            --purple: {PALETTE["purple"]};
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(107, 199, 255, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(90, 228, 168, 0.14), transparent 24%),
                linear-gradient(180deg, #091221 0%, #07111f 58%, #08101a 100%);
            color: var(--text);
        }}

        html, body, [class*="css"] {{
            font-family: "IBM Plex Sans", sans-serif;
        }}

        h1, h2, h3, .hero-title, .section-title {{
            font-family: "Space Grotesk", sans-serif;
            color: var(--text);
        }}

        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(15, 27, 45, 0.98), rgba(8, 17, 31, 0.98));
            border-right: 1px solid rgba(144, 165, 196, 0.14);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--text);
        }}

        .panel {{
            background: linear-gradient(180deg, rgba(19, 35, 59, 0.92), rgba(11, 20, 35, 0.96));
            border: 1px solid rgba(144, 165, 196, 0.12);
            border-radius: 22px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.24);
        }}

        .hero {{
            position: relative;
            overflow: hidden;
            padding: 1.6rem 1.7rem;
            border-radius: 28px;
            border: 1px solid rgba(144, 165, 196, 0.12);
            background:
                linear-gradient(135deg, rgba(13, 27, 47, 0.95), rgba(19, 39, 59, 0.94)),
                radial-gradient(circle at top right, rgba(90, 228, 168, 0.28), transparent 30%);
            min-height: 240px;
        }}

        .hero:after {{
            content: "";
            position: absolute;
            inset: auto -60px -60px auto;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(123, 140, 255, 0.25), transparent 65%);
            pointer-events: none;
        }}

        .eyebrow {{
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.16rem;
            font-size: 0.72rem;
            font-weight: 700;
        }}

        .hero-title {{
            font-size: clamp(2rem, 4vw, 3.7rem);
            line-height: 1;
            margin: 0.45rem 0 0.7rem 0;
        }}

        .hero-copy, .muted {{
            color: var(--muted);
        }}

        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.15rem;
        }}

        .stat-card {{
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(144, 165, 196, 0.1);
            backdrop-filter: blur(10px);
        }}

        .stat-label {{
            color: var(--muted);
            font-size: 0.78rem;
            margin-bottom: 0.35rem;
        }}

        .stat-value {{
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--text);
        }}

        .stat-delta.up {{ color: var(--accent); }}
        .stat-delta.down {{ color: var(--danger); }}
        .stat-delta.flat {{ color: var(--warning); }}

        .section-title {{
            font-size: 1.1rem;
            margin-bottom: 0.75rem;
        }}

        .pill-row {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 0.85rem;
        }}

        .pill {{
            padding: 0.4rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(144, 165, 196, 0.12);
            background: rgba(255, 255, 255, 0.04);
            color: var(--text);
            font-size: 0.8rem;
        }}

        .signal-card {{
            border-radius: 22px;
            padding: 1rem 1rem 0.9rem 1rem;
            background: linear-gradient(180deg, rgba(14, 25, 42, 0.96), rgba(9, 17, 29, 0.96));
            border: 1px solid rgba(144, 165, 196, 0.12);
            min-height: 100%;
        }}

        .signal-head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.9rem;
        }}

        .signal-symbol {{
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
        }}

        .buy {{ background: rgba(90, 228, 168, 0.14); color: var(--accent); }}
        .hold {{ background: rgba(255, 180, 84, 0.14); color: var(--warning); }}
        .sell {{ background: rgba(255, 107, 122, 0.14); color: var(--danger); }}

        .mini-value {{
            font-size: 1.5rem;
            font-family: "Space Grotesk", sans-serif;
            font-weight: 700;
        }}

        .callout {{
            padding: 1rem 1.1rem;
            border-left: 3px solid var(--accent-alt);
            border-radius: 16px;
            background: rgba(107, 199, 255, 0.08);
            color: var(--text);
            margin-top: 0.7rem;
        }}

        .stSelectbox label, .stMultiSelect label, .stSlider label,
        .stDateInput label, .stNumberInput label, .stTextInput label {{
            color: var(--muted) !important;
            font-size: 0.85rem !important;
        }}

        .stButton > button {{
            border-radius: 14px;
            border: 1px solid rgba(90, 228, 168, 0.2);
            background: linear-gradient(135deg, rgba(90, 228, 168, 0.2), rgba(107, 199, 255, 0.18));
            color: var(--text);
            font-weight: 600;
            padding: 0.7rem 1rem;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px 12px 0 0;
            padding: 0.55rem 0.85rem;
        }}

        .stTabs [aria-selected="true"] {{
            background: rgba(123, 140, 255, 0.15);
        }}

        [data-testid="stDataFrame"], [data-testid="stMetric"] {{
            border-radius: 18px;
        }}

        @media (max-width: 900px) {{
            .stat-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(title=None, height=330):
    return dict(
        title=title,
        height=height,
        margin=dict(l=8, r=8, t=40 if title else 8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="IBM Plex Sans"),
        xaxis=dict(showgrid=False, zeroline=False, color=PALETTE["muted"]),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(144, 165, 196, 0.10)",
            zeroline=False,
            color=PALETTE["muted"],
        ),
        hoverlabel=dict(
            bgcolor=PALETTE["panel_alt"],
            bordercolor=PALETTE["stroke"],
            font_color=PALETTE["text"],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01),
    )


def render_panel_title(title, subtitle=""):
    st.markdown(
        f"""
        <div class="panel">
            <div class="section-title">{title}</div>
            <div class="muted">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(app_state):
    equity = app_state["equity_curve"]
    total_return = (equity[-1] / equity[0] - 1) * 100
    sharpe = 1.84
    hit_rate = 64.2
    exposure = app_state["exposure_curve"][-1]
    run_state = "Live monitoring" if st.session_state.trading_active else "Research mode"

    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Neural command center</div>
            <div class="hero-title">Designed like a trading cockpit, not a demo page.</div>
            <div class="hero-copy">
                This workspace blends model telemetry, execution context, and risk posture into one visual system.
                The interface now prioritizes signal clarity, hierarchy, and atmosphere.
            </div>
            <div class="pill-row">
                <div class="pill">Session: {run_state}</div>
                <div class="pill">Universe: {", ".join(app_state["watchlist"][:4])}</div>
                <div class="pill">Refresh profile: 5 minute cadence</div>
            </div>
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-label">Net performance</div>
                    <div class="stat-value">{total_return:,.1f}%</div>
                    <div class="stat-delta up">Above benchmark by 8.4%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Sharpe profile</div>
                    <div class="stat-value">{sharpe:.2f}</div>
                    <div class="stat-delta up">Risk-adjusted edge improving</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Hit rate</div>
                    <div class="stat-value">{hit_rate:.1f}%</div>
                    <div class="stat-delta flat">Selective entries favored</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Capital deployed</div>
                    <div class="stat-value">{exposure:.0f}%</div>
                    <div class="stat-delta down">Sizing constrained by risk engine</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(config, app_state):
    pages = {
        "Overview": "overview",
        "Research": "research",
        "Models": "models",
        "Execution": "execution",
        "Portfolio": "portfolio",
        "Risk": "risk",
        "History": "history",
        "Settings": "settings",
    }

    current_index = list(pages.values()).index(st.session_state.current_page)

    with st.sidebar:
        st.markdown("### NeuralStockTrader")
        st.caption("Quant interface redesign")
        page_label = st.radio("Workspace", list(pages.keys()), index=current_index, label_visibility="collapsed")
        st.session_state.current_page = pages[page_label]

        st.markdown("---")
        st.markdown("#### Session Controls")
        st.selectbox("Trading mode", ["Simulation", "Paper", "Live"], index=0)
        st.selectbox("Focus symbol", app_state["watchlist"], index=0)
        st.slider("Risk budget", 5, 100, 42)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start", use_container_width=True):
                st.session_state.trading_active = True
        with col2:
            if st.button("Pause", use_container_width=True):
                st.session_state.trading_active = False

        st.markdown("---")
        st.markdown("#### System Snapshot")
        st.metric("Environment", config.get("system", {}).get("environment", "development").replace("_", " ").title())
        st.metric("Symbols", len(app_state["watchlist"]))
        st.metric("Max position", f'{config.get("risk_management", {}).get("max_position_size", 0.1) * 100:.0f}%')

        st.markdown("---")
        st.markdown("#### Design Notes")
        st.caption(
            "The new UI leans into darker glass panels, stronger typographic hierarchy, and dashboard density without losing readability."
        )


def render_signal_cards(app_state):
    symbol_states = [
        ("NVDA", "Buy", "0.91 confidence", "+3.4% expected swing"),
        ("MSFT", "Hold", "0.68 confidence", "Awaiting catalyst"),
        ("AAPL", "Buy", "0.83 confidence", "Quality mean re-entry"),
    ]
    cols = st.columns(3)
    for col, (symbol, signal, confidence, note) in zip(cols, symbol_states):
        with col:
            badge_class = {"Buy": "buy", "Hold": "hold", "Sell": "sell"}[signal]
            price = app_state["market"][symbol]["close"].iloc[-1]
            st.markdown(
                f"""
                <div class="signal-card">
                    <div class="signal-head">
                        <div class="signal-symbol">{symbol}</div>
                        <div class="badge {badge_class}">{signal}</div>
                    </div>
                    <div class="mini-value">${price:,.2f}</div>
                    <div class="muted" style="margin-top:0.25rem;">{confidence}</div>
                    <div class="callout">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_overview(app_state):
    render_hero(app_state)
    st.markdown("")
    render_signal_cards(app_state)

    left, right = st.columns([1.8, 1.05], gap="large")

    with left:
        st.markdown('<div class="section-title">Equity curve vs benchmark</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=app_state["equity_dates"],
                y=app_state["equity_curve"],
                mode="lines",
                name="Strategy",
                line=dict(color=PALETTE["accent"], width=3),
                fill="tozeroy",
                fillcolor="rgba(90, 228, 168, 0.12)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=app_state["equity_dates"],
                y=app_state["benchmark_curve"],
                mode="lines",
                name="Benchmark",
                line=dict(color=PALETTE["accent_alt"], width=2, dash="dot"),
            )
        )
        fig.update_layout(**chart_layout(height=360))
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">Open positions</div>', unsafe_allow_html=True)
            df = app_state["positions"][["Symbol", "Theme", "Market Value", "PnL", "PnL %"]].copy()
            df["Market Value"] = df["Market Value"].map(lambda value: f"${value:,.0f}")
            df["PnL"] = df["PnL"].map(lambda value: f"${value:,.0f}")
            df["PnL %"] = df["PnL %"].map(lambda value: f"{value:,.2f}%")
            st.dataframe(df, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown('<div class="section-title">Autonomy modules</div>', unsafe_allow_html=True)
            st.dataframe(app_state["agents"], use_container_width=True, hide_index=True)

    with right:
        st.markdown('<div class="section-title">Capital deployment</div>', unsafe_allow_html=True)
        allocation = pd.DataFrame(
            {
                "Bucket": ["Core longs", "Tactical AI", "Hedges", "Cash"],
                "Value": [46, 24, 12, 18],
            }
        )
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=allocation["Bucket"],
                    values=allocation["Value"],
                    hole=0.62,
                    marker=dict(
                        colors=[
                            PALETTE["accent"],
                            PALETTE["purple"],
                            PALETTE["warning"],
                            PALETTE["accent_alt"],
                        ]
                    ),
                    textinfo="label+percent",
                )
            ]
        )
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=PALETTE["text"], family="IBM Plex Sans"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Recent decisions</div>', unsafe_allow_html=True)
        st.dataframe(app_state["trades"], use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="callout">
                The UI now emphasizes what matters first: confidence, positioning, regime, and execution context.
                The old flat dashboard has been replaced with a clearer trading narrative.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_research(app_state):
    st.markdown('<div class="section-title">Research Lab</div>', unsafe_allow_html=True)
    control_a, control_b, control_c = st.columns([1, 1, 1.2])
    with control_a:
        symbol = st.selectbox("Symbol", app_state["watchlist"], index=0)
    with control_b:
        horizon = st.selectbox("Horizon", ["1 month", "3 months", "6 months", "1 year"], index=1)
    with control_c:
        strategies = st.multiselect(
            "Signal stack",
            ["Neural ensemble", "Momentum", "Mean reversion", "Pairs", "Macro overlay"],
            default=["Neural ensemble", "Momentum"],
        )

    data = app_state["market"][symbol]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["close"],
            mode="lines",
            name=f"{symbol} price",
            line=dict(color=PALETTE["accent_alt"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["close"].rolling(20).mean(),
            mode="lines",
            name="20D trend",
            line=dict(color=PALETTE["warning"], width=2),
        )
    )
    fig.update_layout(**chart_layout(title=f"{symbol} research view", height=360))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([1.2, 1], gap="large")
    with col1:
        factor = pd.DataFrame(
            {
                "Factor": ["Quality", "Momentum", "Volatility", "Crowding", "Macro beta"],
                "Score": [84, 77, 58, 39, 62],
            }
        )
        fig = go.Figure(
            data=[go.Bar(
                x=factor["Score"],
                y=factor["Factor"],
                orientation="h",
                marker=dict(color=[PALETTE["accent"], PALETTE["accent_alt"], PALETTE["purple"], PALETTE["danger"], PALETTE["warning"]]),
            )]
        )
        fig.update_layout(**chart_layout(title="Factor stack", height=300))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">Research memo</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="panel">
                <div class="muted">
                    Horizon selected: {horizon}. Active models: {", ".join(strategies)}.
                    Price structure remains constructive with improving trend persistence and manageable crowding.
                    The new page groups exploratory inputs and visual evidence together instead of scattering them across generic controls.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_models(config):
    training = config.get("neural_network", {}).get("training", {})
    architecture = config.get("neural_network", {}).get("architecture", "lstm").upper()
    st.markdown('<div class="section-title">Model Operations</div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        model_type = st.selectbox("Architecture", ["LSTM", "GRU", "TRANSFORMER", "ENSEMBLE"], index=0)
        epochs = st.slider("Epochs", 20, 200, int(training.get("epochs", 100)))
        batch = st.slider("Batch size", 8, 128, int(training.get("batch_size", 32)))
        lr = st.select_slider("Learning rate", options=[0.0001, 0.0005, 0.001, 0.005], value=0.001)
        if st.button("Launch training cycle", use_container_width=True):
            st.success(f"Queued {model_type} training run at learning rate {lr}.")

        st.markdown(
            f"""
            <div class="callout">
                Default architecture in config: {architecture}. Early stopping is set to
                {str(training.get("early_stopping", True)).lower()} with patience {training.get("patience", 10)}.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        epochs_range = np.arange(1, epochs + 1)
        train_loss = 0.7 * np.exp(-epochs_range / (epochs / 5)) + 0.02
        val_loss = train_loss + np.sin(np.linspace(0, 3, epochs)) * 0.025 + 0.02
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs_range, y=train_loss, mode="lines", name="Train", line=dict(color=PALETTE["accent"], width=3)))
        fig.add_trace(go.Scatter(x=epochs_range, y=val_loss, mode="lines", name="Validation", line=dict(color=PALETTE["warning"], width=2)))
        fig.update_layout(**chart_layout(title="Loss trajectory", height=360))
        st.plotly_chart(fig, use_container_width=True)


def render_execution(app_state):
    st.markdown('<div class="section-title">Execution Theater</div>', unsafe_allow_html=True)
    top_a, top_b, top_c, top_d = st.columns(4)
    top_a.metric("Router quality", "A-", "Latency down 18 ms")
    top_b.metric("Orders today", "12", "4 adaptive slices")
    top_c.metric("Slippage", "4.8 bps", "-1.3 bps")
    top_d.metric("Trading state", "Active" if st.session_state.trading_active else "Paused")

    symbol = st.selectbox("Tape", app_state["watchlist"], index=0, key="execution_symbol")
    tape = app_state["market"][symbol].tail(72)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tape["date"], y=tape["close"], mode="lines", name=symbol, line=dict(color=PALETTE["purple"], width=3)))
    fig.add_trace(go.Bar(x=tape["date"], y=tape["volume"], name="Volume", marker_color="rgba(107,199,255,0.18)", yaxis="y2"))
    fig.update_layout(
        **chart_layout(title=f"{symbol} tape and participation", height=360),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, color=PALETTE["muted"]),
        barmode="overlay",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(app_state["trades"], use_container_width=True, hide_index=True)


def render_portfolio(app_state):
    st.markdown('<div class="section-title">Portfolio Composition</div>', unsafe_allow_html=True)
    positions = app_state["positions"].copy()
    positions["Allocation %"] = positions["Market Value"] / positions["Market Value"].sum() * 100

    col1, col2 = st.columns([1.2, 1], gap="large")
    with col1:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=positions["Symbol"],
                    y=positions["Allocation %"],
                    marker_color=[PALETTE["accent"], PALETTE["accent_alt"], PALETTE["purple"], PALETTE["warning"]],
                )
            ]
        )
        fig.update_layout(**chart_layout(title="Allocation by position", height=320))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">Holdings ledger</div>', unsafe_allow_html=True)
        ledger = positions[["Symbol", "Theme", "Qty", "Market Value", "PnL %"]].copy()
        ledger["Market Value"] = ledger["Market Value"].map(lambda value: f"${value:,.0f}")
        ledger["PnL %"] = ledger["PnL %"].map(lambda value: f"{value:,.2f}%")
        st.dataframe(ledger, use_container_width=True, hide_index=True)


def render_risk(config, app_state):
    risk = config.get("risk_management", {})
    st.markdown('<div class="section-title">Risk Architecture</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Max drawdown guard", f'{risk.get("max_drawdown", 0.2) * 100:.0f}%')
    col2.metric("Daily loss cap", f'{risk.get("max_daily_loss", 0.05) * 100:.1f}%')
    col3.metric("Correlation limit", f'{risk.get("correlation_limit", 0.7):.2f}')

    exposure = pd.DataFrame(
        {
            "Day": app_state["equity_dates"].tail(30),
            "Exposure": app_state["exposure_curve"].tail(30),
        }
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=exposure["Day"], y=exposure["Exposure"], mode="lines", fill="tozeroy", line=dict(color=PALETTE["danger"], width=3)))
    fig.update_layout(**chart_layout(title="Risk budget utilization", height=320))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="callout">
            The risk page now reads like a control room: hard limits up top, utilization trend in the center,
            and fewer generic widgets competing for attention.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history(app_state):
    st.markdown('<div class="section-title">Performance History</div>', unsafe_allow_html=True)
    months = app_state["monthly_returns"]

    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig = go.Figure(data=[go.Bar(x=months["Month"], y=months["Return"], marker_color=PALETTE["accent_alt"])])
        fig.update_layout(**chart_layout(title="Monthly returns", height=320))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cumulative = np.cumprod(1 + months["Return"] / 100)
        fig = go.Figure(data=[go.Scatter(x=months["Month"], y=cumulative, mode="lines+markers", line=dict(color=PALETTE["accent"], width=3))])
        fig.update_layout(**chart_layout(title="Cumulative curve", height=320))
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(app_state["trades"], use_container_width=True, hide_index=True)


def render_settings(config):
    st.markdown('<div class="section-title">Workspace Settings</div>', unsafe_allow_html=True)
    tabs = st.tabs(["General", "Connectivity", "Alerts"])

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.multiselect("Tracked symbols", ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN"], default=config.get("data", {}).get("symbols", []))
            st.selectbox("Timeframe", ["5m", "15m", "1h", "4h", "1d"], index=2)
        with col2:
            st.selectbox("Position sizing", ["Kelly", "Proportional", "Risk parity"], index=0)
            st.slider("Dashboard refresh", 1, 30, 5)
        st.button("Save workspace settings", use_container_width=True)

    with tabs[1]:
        st.text_input("Broker endpoint", value=config.get("api", {}).get("alpaca", {}).get("base_url", ""))
        st.text_input("Broker API key", type="password")
        st.button("Test connectivity", use_container_width=True)

    with tabs[2]:
        st.checkbox("Execution alerts", value=True)
        st.checkbox("Risk alerts", value=True)
        st.checkbox("Daily digest", value=False)
        st.button("Save alerts", use_container_width=True)


def main():
    config = load_config()
    app_state = build_app_state(config)
    inject_styles()
    render_sidebar(config, app_state)

    if st.session_state.current_page == "overview":
        render_overview(app_state)
    elif st.session_state.current_page == "research":
        render_research(app_state)
    elif st.session_state.current_page == "models":
        render_models(config)
    elif st.session_state.current_page == "execution":
        render_execution(app_state)
    elif st.session_state.current_page == "portfolio":
        render_portfolio(app_state)
    elif st.session_state.current_page == "risk":
        render_risk(config, app_state)
    elif st.session_state.current_page == "history":
        render_history(app_state)
    elif st.session_state.current_page == "settings":
        render_settings(config)


if __name__ == "__main__":
    main()
