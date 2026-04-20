"""
Advanced Frontend Components and Utilities
Custom Streamlit components and styling utilities
"""

import streamlit as st
from typing import Dict, List, Optional
import plotly.graph_objects as go
import pandas as pd
import numpy as np


class ThemeConfig:
    """Theme configuration and color schemes"""
    
    # Color Palette
    PRIMARY_GRADIENT = ("667eea", "764ba2")
    SUCCESS_GRADIENT = ("84fab0", "8fd3f4")
    WARNING_GRADIENT = ("fa709a", "fee140")
    
    COLORS = {
        "primary": "#667eea",
        "secondary": "#764ba2",
        "success": "#84fab0",
        "warning": "#fa709a",
        "danger": "#ff6b6b",
        "info": "#4ecdc4",
        "light": "#f7f7f7",
        "dark": "#2c3e50"
    }
    
    @staticmethod
    def apply_theme():
        """Apply theme CSS"""
        st.markdown("""
            <style>
            /* Main container */
            .main {
                padding: 2rem;
                background-color: #f0f2f6;
            }
            
            /* Headers */
            h1, h2, h3 {
                color: #2c3e50;
                font-weight: 600;
            }
            
            h1 {
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }
            
            /* Metric cards */
            [data-testid="metric-container"] {
                background-color: white;
                border-left: 4px solid #667eea;
                border-radius: 10px;
                padding: 1.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            /* Dataframe styling */
            [data-testid="dataframe"] {
                border-radius: 10px;
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            /* Buttons */
            .stButton > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.75rem 1.5rem;
                font-weight: 600;
                box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
            }
            
            /* Input fields */
            .stTextInput > div > div > input,
            .stNumberInput > div > div > input,
            .stSelectbox > div > div > select,
            .stMultiSelect > div > div > div {
                border-radius: 8px;
                border: 2px solid #e0e0e0;
                padding: 0.75rem;
                transition: border-color 0.3s ease;
            }
            
            .stTextInput > div > div > input:focus,
            .stNumberInput > div > div > input:focus,
            .stSelectbox > div > div > select:focus,
            .stMultiSelect > div > div > div:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] button {
                border-radius: 8px 8px 0 0;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: #667eea;
                color: white;
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            /* Custom cards */
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }
            
            .success-card {
                background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                color: #333;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(132, 250, 176, 0.3);
            }
            
            .warning-card {
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                color: #333;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(250, 112, 154, 0.3);
            }
            
            .info-card {
                background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(78, 205, 196, 0.3);
            }
            
            /* Animations */
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .stMetric {
                animation: slideIn 0.5s ease-out;
            }
            </style>
        """, unsafe_allow_html=True)


class CustomComponents:
    """Custom Streamlit components"""
    
    @staticmethod
    def metric_card(title: str, value: str, delta: str = "", icon: str = ""):
        """Create a custom metric card"""
        html = f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">{title}</p>
                    <p style="margin: 0; font-size: 2rem; font-weight: bold;">{value}</p>
                    <p style="margin: 0; font-size: 0.85rem; opacity: 0.8;">{delta}</p>
                </div>
                <div style="font-size: 2.5rem;">{icon}</div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    @staticmethod
    def status_badge(status: str, severity: str = "info"):
        """Create a status badge"""
        color_map = {
            "success": "#84fab0",
            "warning": "#fee140",
            "danger": "#ff6b6b",
            "info": "#4ecdc4"
        }
        color = color_map.get(severity, "#667eea")
        html = f"""
        <span style="
            background-color: {color};
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.85rem;
        ">{status}</span>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    @staticmethod
    def progress_card(title: str, progress: float, color: str = "#667eea"):
        """Create a progress card"""
        html = f"""
        <div style="margin-bottom: 1.5rem;">
            <p style="margin-bottom: 0.5rem; font-weight: bold;">{title}</p>
            <div style="
                background-color: #e0e0e0;
                border-radius: 10px;
                height: 10px;
                overflow: hidden;
            ">
                <div style="
                    background: linear-gradient(90deg, {color} 0%, rgba(0,0,0,0.1) 100%);
                    height: 100%;
                    width: {progress}%;
                    border-radius: 10px;
                    transition: width 0.5s ease;
                "></div>
            </div>
            <p style="margin-top: 0.5rem; text-align: right; font-size: 0.9rem;">{progress:.1f}%</p>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    @staticmethod
    def price_ticker(symbol: str, price: float, change: float, change_pct: float):
        """Create a price ticker widget"""
        change_color = "#84fab0" if change > 0 else "#ff6b6b"
        change_symbol = "▲" if change > 0 else "▼"
        
        html = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="margin: 0; font-size: 1.2rem; font-weight: bold;">{symbol}</p>
                    <p style="margin: 0; font-size: 2rem; font-weight: bold;">${price:.2f}</p>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; font-size: 1rem; color: {change_color};">
                        {change_symbol} ${change:.2f}
                    </p>
                    <p style="margin: 0; font-size: 1rem; color: {change_color};">
                        ({change_pct:+.2f}%)
                    </p>
                </div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


class ChartFactory:
    """Factory for creating common charts"""
    
    @staticmethod
    def create_line_chart(data: pd.DataFrame, x_col: str, y_col: str, 
                         title: str, color: str = "#667eea"):
        """Create a line chart"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode='lines',
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=color.replace(')', ', 0.2)').replace('rgb', 'rgba')
        ))
        
        fig.update_layout(
            title=title,
            height=400,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_showgrid=False,
            yaxis_showgrid=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        return fig
    
    @staticmethod
    def create_bar_chart(data: Dict[str, float], title: str, colors: List[str] = None):
        """Create a bar chart"""
        if colors is None:
            colors = ["#667eea"] * len(data)
        
        fig = go.Figure(data=[
            go.Bar(x=list(data.keys()), y=list(data.values()), marker_color=colors)
        ])
        
        fig.update_layout(
            title=title,
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_showgrid=False,
            yaxis_showgrid=True
        )
        return fig
    
    @staticmethod
    def create_pie_chart(data: Dict[str, float], title: str, 
                         colors: List[str] = None):
        """Create a pie chart"""
        if colors is None:
            colors = ["#667eea", "#764ba2", "#84fab0", "#fa709a", "#fee140"]
        
        fig = go.Figure(data=[
            go.Pie(labels=list(data.keys()), values=list(data.values()),
                   marker=dict(colors=colors[:len(data)]))
        ])
        
        fig.update_layout(
            title=title,
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        return fig
    
    @staticmethod
    def create_candlestick_chart(data: pd.DataFrame, title: str):
        """Create a candlestick chart"""
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )
        ])
        
        fig.update_layout(
            title=title,
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False
        )
        return fig


class DataFormatter:
    """Utilities for formatting data for display"""
    
    @staticmethod
    def format_currency(value: float) -> str:
        """Format as currency"""
        return f"${value:,.2f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format as percentage"""
        return f"{value:+.{decimals}f}%"
    
    @staticmethod
    def format_number(value: float, decimals: int = 2) -> str:
        """Format as number"""
        return f"{value:,.{decimals}f}"
    
    @staticmethod
    def get_change_color(value: float) -> str:
        """Get color based on change value"""
        if value > 0:
            return "#84fab0"
        elif value < 0:
            return "#ff6b6b"
        else:
            return "#gray"
    
    @staticmethod
    def get_status_icon(status: str) -> str:
        """Get icon for status"""
        icons = {
            "success": "✅",
            "pending": "⏳",
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️",
            "buy": "🟢",
            "sell": "🔴",
            "hold": "🟡"
        }
        return icons.get(status, "•")


class StateManager:
    """Manage session state"""
    
    @staticmethod
    def initialize_states():
        """Initialize all session states"""
        defaults = {
            "current_page": "dashboard",
            "trading_active": False,
            "selected_symbol": "AAPL",
            "selected_strategy": "ML Ensemble",
            "backtest_results": None,
            "training_results": None,
            "theme": "light"
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def get_state(key: str, default=None):
        """Get session state value"""
        return st.session_state.get(key, default)
    
    @staticmethod
    def set_state(key: str, value):
        """Set session state value"""
        st.session_state[key] = value
    
    @staticmethod
    def update_state(key: str, callback):
        """Update state with callback"""
        st.session_state[key] = callback(st.session_state.get(key))


# Utility functions
def show_success_message(message: str):
    """Display success message"""
    st.success(f"✅ {message}")


def show_error_message(message: str):
    """Display error message"""
    st.error(f"❌ {message}")


def show_warning_message(message: str):
    """Display warning message"""
    st.warning(f"⚠️ {message}")


def show_info_message(message: str):
    """Display info message"""
    st.info(f"ℹ️ {message}")


def create_two_column_layout():
    """Create a two-column layout"""
    return st.columns(2)


def create_three_column_layout():
    """Create a three-column layout"""
    return st.columns(3)


def create_custom_divider():
    """Create a custom divider"""
    st.markdown("""
    <hr style="
        border: none;
        border-top: 2px solid;
        border-image: linear-gradient(to right, #667eea, #764ba2) 1;
        margin: 1.5rem 0;
    ">
    """, unsafe_allow_html=True)
