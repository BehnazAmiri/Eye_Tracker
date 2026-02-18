"""
Simple Models Package
====================
Modular deep learning and machine learning models for eye-tracking classification.
"""

from .mlp import SimpleMLP
from .cnn import CNN1D
from .lstm import SimpleLSTM
from .hybrid_cnn_lstm import HybridCNNLSTM
from .transformer import SimpleTransformer
from .ml_models import create_ml_model, get_model_info, HAS_XGBOOST

__all__ = [
    'SimpleMLP', 
    'CNN1D', 
    'SimpleLSTM', 
    'HybridCNNLSTM',
    'SimpleTransformer',
    'create_ml_model',
    'get_model_info',
    'HAS_XGBOOST'
]


def create_model(model_type, config):
    """
    Factory function to create models.
    
    Args:
        model_type: 'mlp', 'cnn', 'cnn1d', 'lstm', 'bilstm', 'hybrid', or 'transformer'
        config: Model configuration dictionary
    
    Returns:
        Model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'mlp':
        return SimpleMLP(
            input_dim=config.get('input_dim', 50),
            hidden_dims=config.get('hidden_dims', [64, 32]),
            dropout=config.get('dropout', 0.3),
            use_batch_norm=config.get('use_batch_norm', True),
            num_classes=1  # Binary classification for BCEWithLogitsLoss
        )
    
    elif model_type == 'cnn':
        return CNN1D(
            input_channels=config.get('input_channels', 8),
            num_conv_layers=config.get('num_conv_layers', 2),
            base_channels=config.get('base_channels', 16),
            kernel_size=config.get('kernel_size', 5),
            dropout=config.get('dropout', 0.3),
            fc_hidden_dim=config.get('fc_hidden_dim', 16),
            num_classes=1  # Binary classification
        )
    
    elif model_type == 'lstm':
        return SimpleLSTM(
            input_size=config.get('input_size', 8),
            hidden_size=config.get('hidden_size', 32),
            num_layers=config.get('num_layers', 1),
            dropout=config.get('dropout', 0.3),
            fc_hidden_dim=config.get('fc_hidden_dim', 16),
            bidirectional=config.get('bidirectional', False),
            num_classes=1  # Binary classification
        )
    
    elif model_type == 'bilstm':
        return SimpleLSTM(
            input_size=config.get('input_size', 8),
            hidden_size=config.get('hidden_size', 32),
            num_layers=config.get('num_layers', 1),
            dropout=config.get('dropout', 0.3),
            fc_hidden_dim=config.get('fc_hidden_dim', 16),
            bidirectional=config.get('bidirectional', True),
            num_classes=1  # Binary classification
        )
    
    elif model_type == 'cnn1d':
        return CNN1D(
            input_channels=config.get('input_channels', 8),
            num_conv_layers=config.get('num_conv_layers', 2),
            base_channels=config.get('base_channels', 16),
            kernel_size=config.get('kernel_size', 5),
            dropout=config.get('dropout', 0.3),
            fc_hidden_dim=config.get('fc_hidden_dim', 16),
            num_classes=1  # Binary classification
        )
    
    elif model_type == 'hybrid':
        return HybridCNNLSTM(
            input_channels=config.get('input_channels', 8),
            cnn_channels=config.get('cnn_channels', 32),
            cnn_kernel_size=config.get('cnn_kernel_size', 5),
            lstm_hidden_size=config.get('lstm_hidden_size', 32),
            lstm_num_layers=config.get('lstm_num_layers', 1),
            dropout=config.get('dropout', 0.3),
            fc_hidden_dim=config.get('fc_hidden_dim', 16),
            num_classes=1  # Binary classification
        )
    
    elif model_type == 'transformer':
        return SimpleTransformer(
            input_size=config.get('input_size', 8),
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 4),
            dim_feedforward=config.get('dim_feedforward', 512),
            dropout=config.get('dropout', 0.3),
            fc_hidden_dim=config.get('fc_hidden_dim', 64),
            max_seq_len=config.get('sequence_length', 500),
            num_classes=1  # Binary classification
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: mlp, cnn, cnn1d, lstm, bilstm, hybrid, transformer")
