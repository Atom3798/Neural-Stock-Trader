"""
Data flow and feature engineering visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def visualize_data_flow():
    """Visualize how data flows through the system"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'NeuralStockTrader - Data Pipeline & Feature Engineering', 
            fontsize=18, fontweight='bold', ha='center')
    
    colors = {
        'data_source': '#FF6B6B',
        'preprocessing': '#FFA07A',
        'features': '#FFE66D',
        'model': '#4ECDC4',
        'output': '#95E1D3',
        'evaluation': '#45B7D1'
    }
    
    # 1. Data Source
    box = FancyBboxPatch((0.5, 9.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=colors['data_source'], linewidth=2)
    ax.add_patch(box)
    ax.text(1.75, 10.3, 'DATA SOURCE', ha='center', fontweight='bold', fontsize=10)
    ax.text(1.75, 9.8, 'yfinance API\nHistorical OHLCV\n250 days data', 
            ha='center', fontsize=8)
    
    # Arrow
    arrow = FancyArrowPatch((3, 10.1), (4.5, 10.1), arrowstyle='->', mutation_scale=25,
                           linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # 2. Raw Data Processing
    box = FancyBboxPatch((4.5, 9.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=colors['preprocessing'], linewidth=2)
    ax.add_patch(box)
    ax.text(5.75, 10.3, 'PREPROCESSING', ha='center', fontweight='bold', fontsize=10)
    ax.text(5.75, 9.8, 'Remove NaN\nNormalize OHLC\nHandle Outliers', 
            ha='center', fontsize=8)
    
    # Arrow
    arrow = FancyArrowPatch((7, 10.1), (8.5, 10.1), arrowstyle='->', mutation_scale=25,
                           linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # 3. Feature Engineering (Left branch)
    box = FancyBboxPatch((8.5, 9.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=colors['features'], linewidth=2)
    ax.add_patch(box)
    ax.text(9.75, 10.3, 'FEATURE ENG.', ha='center', fontweight='bold', fontsize=10)
    ax.text(9.75, 9.8, 'Technical Indicators\nPrice Patterns\nVolume Analysis', 
            ha='center', fontsize=8)
    
    # 3.1 Technical Indicators Detail
    y_pos = 8.8
    indicators = [
        '• SMA (5, 20, 50, 200)',
        '• RSI (14)',
        '• MACD',
        '• Bollinger Bands',
        '• ATR',
        '• Volume Indicators',
        '• Returns/Momentum',
        '• Price Patterns'
    ]
    
    indicator_box = FancyBboxPatch((8, 6.5), 3.5, 2.5, boxstyle="round,pad=0.1",
                                   edgecolor='#333', facecolor='#fff9e6', 
                                   linewidth=1.5, linestyle='--', alpha=0.8)
    ax.add_patch(indicator_box)
    ax.text(9.75, 8.7, 'Technical Indicators', ha='center', fontweight='bold', fontsize=9)
    
    for i, indicator in enumerate(indicators):
        ax.text(8.3, 8.3 - i*0.25, indicator, ha='left', fontsize=7, family='monospace')
    
    # Arrow from feature eng down
    arrow = FancyArrowPatch((9.75, 9.5), (9.75, 9.0), arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black')
    ax.add_patch(arrow)
    
    # 4. Feature Scaling & Sequencing
    arrow = FancyArrowPatch((9.75, 6.5), (9.75, 5.8), arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    box = FancyBboxPatch((8, 5), 3.5, 0.8, boxstyle="round,pad=0.08",
                         edgecolor='black', facecolor='#FFD6A5', linewidth=2)
    ax.add_patch(box)
    ax.text(9.75, 5.4, 'Normalization & Sequencing (window=50)', 
            ha='center', fontweight='bold', fontsize=9)
    
    # Arrow down to model input
    arrow = FancyArrowPatch((9.75, 5), (9.75, 4.2), arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # Right branch - Price Target
    arrow = FancyArrowPatch((11, 10.1), (12.5, 10.1), arrowstyle='->', mutation_scale=25,
                           linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    box = FancyBboxPatch((12.5, 9.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=colors['output'], linewidth=2)
    ax.add_patch(box)
    ax.text(13.75, 10.3, 'TARGET LABEL', ha='center', fontweight='bold', fontsize=10)
    ax.text(13.75, 9.8, 'Next Day Return\nDirection (Up/Down)\nPrice Target', 
            ha='center', fontsize=8)
    
    # Arrow down to model
    arrow = FancyArrowPatch((13.75, 9.5), (13.75, 4.8), arrowstyle='->', mutation_scale=25,
                           linewidth=2.5, color='black', linestyle='--')
    ax.add_patch(arrow)
    
    # 5. Neural Network Model
    box = FancyBboxPatch((7, 2.5), 5.5, 1.5, boxstyle="round,pad=0.12",
                         edgecolor='black', facecolor=colors['model'], linewidth=2.5)
    ax.add_patch(box)
    ax.text(9.75, 3.7, 'LSTM NEURAL NETWORK', ha='center', fontweight='bold', fontsize=11)
    ax.text(9.75, 3.2, 'Input: 50-step sequences × 50 features\nOutput: Price prediction (0-1 normalized)', 
            ha='center', fontsize=8)
    
    # Arrow input
    arrow = FancyArrowPatch((9.75, 4.2), (9.75, 4.0), arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color='black')
    ax.add_patch(arrow)
    ax.text(10.3, 4.1, 'Input Data\n(50, 50)', fontsize=7, style='italic')
    
    # Arrow output
    arrow = FancyArrowPatch((9.75, 2.5), (9.75, 1.8), arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color='black')
    ax.add_patch(arrow)
    ax.text(10.3, 2.1, 'Prediction\n(batch, 1)', fontsize=7, style='italic')
    
    # 6. Post-Processing
    box = FancyBboxPatch((7.5, 0.8), 4.5, 0.9, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=colors['evaluation'], linewidth=2)
    ax.add_patch(box)
    ax.text(9.75, 1.5, 'POST-PROCESSING & TRADING SIGNALS', 
            ha='center', fontweight='bold', fontsize=10)
    ax.text(9.75, 1.05, 'Denormalize • Generate Signals • Risk Management', 
            ha='center', fontsize=8)
    
    # Left side - Training Process
    ax.text(2, 7.5, 'TRAINING PROCESS', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFE66D', alpha=0.5))
    
    train_text = """
    1. Train/Val Split (80/20)
    2. Batch Size: 32
    3. Epochs: 100
    4. Optimizer: Adam (lr=0.001)
    5. Loss: MSE
    6. EarlyStopping: patience=10
    7. ReduceLROnPlateau
    """
    
    ax.text(0.5, 6.8, train_text, fontsize=7.5, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.8, pad=0.5))
    
    # Bottom - Feature Statistics
    ax.text(2, 0.3, 'Input Shape: (n_samples, 50_timesteps, 50_features) | Output Shape: (n_samples, 1)', 
            fontsize=9, style='italic', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Data flow diagram saved as: data_flow_diagram.png")
    plt.show()


def visualize_feature_importance():
    """Visualize which features matter most"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('NeuralStockTrader - Feature Analysis', fontsize=16, fontweight='bold')
    
    # Feature importance (simulated)
    features = ['RSI', 'MACD', 'SMA20', 'Momentum', 'Volume', 'ATR', 'Bollinger', 
                'Returns', 'SMA50', 'Price', 'VolumeMA', 'Beta']
    importance = np.array([95, 88, 82, 78, 75, 72, 68, 65, 60, 58, 55, 50])
    
    ax = axes[0]
    colors_bar = plt.cm.RdYlGn(importance / 100)
    bars = ax.barh(features, importance, color=colors_bar, edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax.set_title('Feature Importance in LSTM Model', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 100])
    for i, (feature, score) in enumerate(zip(features, importance)):
        ax.text(score + 1, i, f'{score}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    # Feature categories
    ax = axes[1]
    ax.axis('off')
    
    categories = """
    FEATURE CATEGORIES
    
    Momentum Indicators:
      • RSI (Relative Strength Index)
      • MACD (Moving Avg Convergence/Div)
      • Momentum Oscillator
      • Beta
    
    Trend Indicators:
      • SMA 20, 50, 200 (Simple Moving Avgs)
      • Bollinger Bands
      • ATR (Average True Range)
    
    Volume Indicators:
      • Volume
      • Volume Moving Average
    
    Price Actions:
      • Close/Open/High/Low
      • Returns (daily %)
    
    Each feature is normalized (0-1) before
    feeding into the LSTM network
    """
    
    ax.text(0.05, 0.95, categories, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, pad=0.8))
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance chart saved as: feature_importance.png")
    plt.show()


if __name__ == "__main__":
    print("Generating data flow visualizations...\n")
    
    print("Creating data pipeline diagram...")
    visualize_data_flow()
    
    print("\n\nCreating feature analysis...")
    visualize_feature_importance()
    
    print("\n✓ All data flow visualizations completed!")
