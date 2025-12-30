"""
Visualize the neural network architecture
"""

import torch
import torch.nn as nn
from src.model_layer.neural_networks import LSTMModel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def visualize_lstm_architecture():
    """Create a visual representation of the LSTM model architecture"""
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    color_input = '#FF6B6B'
    color_lstm = '#4ECDC4'
    color_fc = '#45B7D1'
    color_output = '#FFA07A'
    
    # Title
    ax.text(5, 9.5, 'NeuralStockTrader - LSTM Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Input Layer
    input_box = FancyBboxPatch((0.5, 7.5), 1.5, 1, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=color_input, 
                               linewidth=2, alpha=0.7)
    ax.add_patch(input_box)
    ax.text(1.25, 8, 'Input\n(Sequence)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(1.25, 7.1, 'Shape: (batch, seq_len, 50)', ha='center', 
            fontsize=9, style='italic')
    
    # LSTM Layers
    lstm_y = 5.5
    for i in range(2):
        lstm_box = FancyBboxPatch((2.5 + i*2, lstm_y - i*0.5), 1.8, 1.2,
                                  boxstyle="round,pad=0.1",
                                  edgecolor='black', facecolor=color_lstm,
                                  linewidth=2, alpha=0.7)
        ax.add_patch(lstm_box)
        ax.text(3.4 + i*2, lstm_y - i*0.5 + 0.6, f'LSTM Layer {i+1}', 
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(3.4 + i*2, lstm_y - i*0.5 + 0.2, 'hidden: 128', 
                ha='center', va='center', fontsize=8)
    
    # Arrows from input to first LSTM
    arrow1 = FancyArrowPatch((2, 8), (2.5, 6.1),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Arrows between LSTM layers
    arrow2 = FancyArrowPatch((4.3, 5.5), (4.5, 5),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Dropout Layer
    dropout_box = FancyBboxPatch((2.5, 3), 1.8, 0.8,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='black', facecolor='#FFE66D',
                                 linewidth=2, alpha=0.7)
    ax.add_patch(dropout_box)
    ax.text(3.4, 3.4, 'Dropout\n(0.2)', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Arrow from LSTM to Dropout
    arrow3 = FancyArrowPatch((6.3, 4.5), (3.4, 3.8),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # Fully Connected Layer
    fc_box = FancyBboxPatch((2.5, 1), 1.8, 0.8,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor=color_fc,
                           linewidth=2, alpha=0.7)
    ax.add_patch(fc_box)
    ax.text(3.4, 1.4, 'Dense Layer', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(3.4, 1, '128 -> 1', ha='center', va='center', fontsize=8)
    
    # Arrow from Dropout to FC
    arrow4 = FancyArrowPatch((3.4, 3), (3.4, 1.8),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    # Output Layer
    output_box = FancyBboxPatch((2.5, -0.8), 1.8, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=color_output,
                               linewidth=2, alpha=0.7)
    ax.add_patch(output_box)
    ax.text(3.4, -0.4, 'Output\n(Prediction)', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(3.4, -1.2, 'Shape: (batch, 1)', ha='center', 
            fontsize=9, style='italic')
    
    # Arrow from FC to Output
    arrow5 = FancyArrowPatch((3.4, 1), (3.4, 0),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow5)
    
    # Information panel
    info_text = """
    Model Configuration:
    • Input Size: 50 features
    • Hidden Size: 128 units
    • Num Layers: 2 LSTM layers
    • Dropout: 0.2
    • Output Size: 1 (price prediction)
    • Activation: Tanh (default in LSTM)
    • Optimizer: Adam
    • Loss Function: MSE (Mean Squared Error)
    """
    
    ax.text(6.5, 7, info_text, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input Layer'),
        mpatches.Patch(facecolor=color_lstm, edgecolor='black', label='LSTM Layers'),
        mpatches.Patch(facecolor='#FFE66D', edgecolor='black', label='Regularization'),
        mpatches.Patch(facecolor=color_fc, edgecolor='black', label='Dense Layer'),
        mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output Layer'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('neural_network_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Neural network architecture saved as: neural_network_architecture.png")
    plt.show()


def print_model_summary():
    """Print detailed model summary"""
    print("\n" + "="*60)
    print("LSTM MODEL SUMMARY")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=50, hidden_size=128, num_layers=2, output_size=1, device=device)
    
    print(f"\nModel Architecture:")
    print(model)
    
    print(f"\n\nModel Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    print(f"\n\nLayer Details:")
    for name, param in model.named_parameters():
        print(f"{name:30s} | Shape: {str(list(param.shape)):20s} | Parameters: {param.numel():,}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("Generating neural network visualizations...\n")
    
    # Print model summary
    print_model_summary()
    
    # Generate architecture diagram
    print("\n\nGenerating architecture diagram...")
    visualize_lstm_architecture()
