"""Plot generation for reports."""

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for thread safety
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Set professional style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
})


def plot_training_curves(
    history: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (12, 8),
    dpi: int = 120
):
    """
    Plot training curves (loss, accuracy, F1).
    
    Args:
        history: Training history dataframe
        output_path: Output file path
        figsize: Figure size
        dpi: DPI for saved figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=100)
    
    # Color palette
    train_color = '#3498db'
    val_color = '#e74c3c'
    
    # Plot loss
    axes[0, 0].plot(history['epoch'], history['train_loss'], 
                    label='Train', color=train_color, marker='o', 
                    markersize=4, linewidth=2, alpha=0.8)
    axes[0, 0].plot(history['epoch'], history['val_loss'], 
                    label='Validation', color=val_color, marker='s', 
                    markersize=4, linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Epoch', fontsize=10)
    axes[0, 0].set_ylabel('Loss', fontsize=10)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot accuracy
    axes[0, 1].plot(history['epoch'], history['train_acc'], 
                    label='Train', color=train_color, marker='o', 
                    markersize=4, linewidth=2, alpha=0.8)
    axes[0, 1].plot(history['epoch'], history['val_acc'], 
                    label='Validation', color=val_color, marker='s', 
                    markersize=4, linewidth=2, alpha=0.8)
    axes[0, 1].set_xlabel('Epoch', fontsize=10)
    axes[0, 1].set_ylabel('Accuracy', fontsize=10)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=11, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Plot F1 score
    axes[1, 0].plot(history['epoch'], history['train_f1_macro'], 
                    label='Train', color=train_color, marker='o', 
                    markersize=4, linewidth=2, alpha=0.8)
    axes[1, 0].plot(history['epoch'], history['val_f1_macro'], 
                    label='Validation', color=val_color, marker='s', 
                    markersize=4, linewidth=2, alpha=0.8)
    axes[1, 0].set_xlabel('Epoch', fontsize=10)
    axes[1, 0].set_ylabel('F1 Score (Macro)', fontsize=10)
    axes[1, 0].set_title('Training and Validation F1 Score', fontsize=11, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot precision and recall
    axes[1, 1].plot(history['epoch'], history['val_precision_macro'], 
                    label='Precision', color='#2ecc71', marker='o', 
                    markersize=4, linewidth=2, alpha=0.8)
    axes[1, 1].plot(history['epoch'], history['val_recall_macro'], 
                    label='Recall', color='#f39c12', marker='s', 
                    markersize=4, linewidth=2, alpha=0.8)
    axes[1, 1].plot(history['epoch'], history['val_f1_macro'], 
                    label='F1', color='#9b59b6', marker='^', 
                    markersize=4, linewidth=2, alpha=0.8)
    axes[1, 1].set_xlabel('Epoch', fontsize=10)
    axes[1, 1].set_ylabel('Score', fontsize=10)
    axes[1, 1].set_title('Validation Metrics', fontsize=11, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    class_names: List[str] = ['NOT_RANDOM', 'RANDOM'],
    figsize: tuple = (8, 6),
    dpi: int = 120
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (numpy array or list)
        output_path: Output file path
        class_names: Class names
        figsize: Figure size
        dpi: DPI for saved figure
    """
    # Convert to numpy array if needed
    if not isinstance(cm, np.ndarray):
        cm = np.array(cm)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # Normalize to percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot heatmap WITHOUT annotations (we'll add custom ones)
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap='RdYlGn_r',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8},
        ax=ax,
        linewidths=3,
        linecolor='white',
        vmin=0,
        vmax=100
    )
    
    # Add custom annotations: COUNT (big) and percentage (small)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Count (large, bold)
            ax.text(
                j + 0.5, i + 0.4,
                f'{int(cm[i, j])}',
                ha="center", va="center",
                color="black", fontsize=32, fontweight='bold'
            )
            # Percentage (smaller, below count)
            ax.text(
                j + 0.5, i + 0.65,
                f'({cm_norm[i, j]:.1f}%)',
                ha="center", va="center",
                color="dimgray", fontsize=16
            )
    
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
    
    # Adjust tick labels
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_class_distribution(
    y: np.ndarray,
    output_path: Path,
    title: str = "Class Distribution",
    class_names: List[str] = ['NOT_RANDOM', 'RANDOM'],
    figsize: tuple = (8, 6),
    dpi: int = 120
):
    """
    Plot class distribution bar chart.
    
    Args:
        y: Labels array
        output_path: Output file path
        title: Plot title
        class_names: Class names
        figsize: Figure size
        dpi: DPI for saved figure
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    unique, counts = np.unique(y, return_counts=True)
    percentages = counts / len(y) * 100
    
    # Professional color palette
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(unique, counts, color=colors, alpha=0.8, 
                   edgecolor='white', linewidth=2)
    
    # Add value labels on bars with better styling
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + max(counts) * 0.02,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )
    
    ax.set_xticks(unique)
    ax.set_xticklabels(class_names, fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(counts) * 1.15)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
