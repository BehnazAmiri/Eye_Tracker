"""
Configuration Loader for Deep Learning Pipeline
===============================================
Centralized configuration loading from config.ini
"""

import configparser
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


class Config:
    """
    Configuration manager for deep learning pipeline.
    
    Usage:
        config = Config()
        print(config.data.time_window_s)
        print(config.training.learning_rate_lstm)
    """
    
    def __init__(self, config_file: str = "config.ini"):
        """
        Load configuration from file.
        
        Args:
            config_file: Path to config file (default: config.ini)
        """
        self.config_file = config_file
        self.base_dir = Path(__file__).parent
        self.config_path = self.base_dir / config_file
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        self.parser = configparser.ConfigParser()
        self.parser.read(self.config_path, encoding='utf-8')
        
        # Load all configuration sections
        self.paths = self._load_paths()
        self.data = self._load_data()
        self.augmentation = self._load_augmentation()
        self.training = self._load_training()
        self.model_mlp = self._load_model_mlp()
        self.model_lstm = self._load_model_lstm()
        self.model_bilstm = self._load_model_bilstm()
        self.model_cnn = self._load_model_cnn()
        self.model_cnn1d = self._load_model_cnn1d()
        self.model_hybrid = self._load_model_hybrid()
        self.model_transformer = self._load_model_transformer()
        self.model_rf = self._load_model_rf()
        self.model_xgboost = self._load_model_xgboost()
    
    def _resolve_path(self, path: str) -> Path:
        """Convert relative path to absolute."""
        p = Path(path)
        if not p.is_absolute():
            p = self.base_dir / p
        return p.resolve()
    
    def _load_paths(self) -> Dict[str, Path]:
        """Load file paths."""
        section = self.parser['PATHS']
        return {
            'labels_csv': self._resolve_path(section.get('labels_csv')),
            'trials_dir': self._resolve_path(section.get('trials_dir')),
            'outputs_dir': self._resolve_path(section.get('outputs_dir')),
            'reports_dir': self._resolve_path(section.get('reports_dir')),
            'dl_inputs_dir': self._resolve_path(section.get('dl_inputs_dir'))
        }
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data configuration."""
        section = self.parser['DATA']
        
        # Parse comma-separated lists
        parts_str = section.get('parts_filter', '').strip()
        parts = [p.strip() for p in parts_str.split(',') if p.strip()] if parts_str else []
        
        aoi_str = section.get('aoi_filter', '').strip()
        aoi = [a.strip() for a in aoi_str.split(',') if a.strip()] if aoi_str else []
        
        # Parse feature columns (13 optimal features)
        # Return None if not present to enable auto-inference of ALL features
        features_str = section.get('feature_columns', '').strip()
        features = [f.strip() for f in features_str.split(',') if f.strip()] if features_str else None
        
        time_window = section.get('time_window_s')
        time_window = int(time_window) if time_window.lower() != 'none' else None
        
        # Statistical features flag
        use_stat_features = section.getboolean('use_statistical_features', fallback=False)
        
        return {
            'parts_filter': parts,
            'aoi_filter': aoi,
            'feature_columns': features,
            'time_window_s': time_window,
            'sampling_rate': section.getint('sampling_rate'),
            'test_size': section.getfloat('test_size'),
            'random_seed': section.getint('random_seed'),
            'use_statistical_features': use_stat_features
        }
    
    def _load_augmentation(self) -> Dict[str, Any]:
        """Load data augmentation configuration."""
        if 'AUGMENTATION' not in self.parser:
            # Return default values if section doesn't exist
            return {
                'use_augmentation': False,
                'augment_factor': 2,
                'augment_methods': ['time_shift', 'noise', 'scale']
            }
        
        section = self.parser['AUGMENTATION']
        
        # Parse comma-separated methods
        methods_str = section.get('augment_methods', 'time_shift,noise,scale').strip()
        methods = [m.strip() for m in methods_str.split(',') if m.strip()]
        
        return {
            'use_augmentation': section.getboolean('use_augmentation', fallback=False),
            'augment_factor': section.getint('augment_factor', fallback=2),
            'augment_methods': methods
        }
    
    def _load_training(self) -> Dict[str, Any]:
        """Load training configuration."""
        section = self.parser['TRAINING']
        return {
            'epochs_mlp': section.getint('epochs_mlp'),
            'epochs_lstm': section.getint('epochs_lstm'),
            'epochs_bilstm': section.getint('epochs_bilstm'),
            'epochs_cnn': section.getint('epochs_cnn'),
            'epochs_cnn1d': section.getint('epochs_cnn1d'),
            'epochs_hybrid': section.getint('epochs_hybrid'),
            'epochs_transformer': section.getint('epochs_transformer'),
            'early_stopping_patience': section.getint('early_stopping_patience'),
            'batch_size_mlp': section.getint('batch_size_mlp'),
            'batch_size_lstm': section.getint('batch_size_lstm'),
            'batch_size_bilstm': section.getint('batch_size_bilstm'),
            'batch_size_cnn': section.getint('batch_size_cnn'),
            'batch_size_cnn1d': section.getint('batch_size_cnn1d'),
            'batch_size_hybrid': section.getint('batch_size_hybrid'),
            'batch_size_transformer': section.getint('batch_size_transformer'),
            'learning_rate_mlp': section.getfloat('learning_rate_mlp'),
            'learning_rate_lstm': section.getfloat('learning_rate_lstm'),
            'learning_rate_bilstm': section.getfloat('learning_rate_bilstm'),
            'learning_rate_cnn': section.getfloat('learning_rate_cnn'),
            'learning_rate_cnn1d': section.getfloat('learning_rate_cnn1d'),
            'learning_rate_hybrid': section.getfloat('learning_rate_hybrid'),
            'learning_rate_transformer': section.getfloat('learning_rate_transformer'),
            'weight_decay': section.getfloat('weight_decay'),
            'pos_weight': section.getfloat('pos_weight', fallback=None),
            'device': section.get('device')
        }
    
    def _load_model_mlp(self) -> Dict[str, Any]:
        """Load MLP model configuration."""
        section = self.parser['MODEL_MLP']
        hidden_dims = [int(x.strip()) for x in section.get('hidden_dims').split(',')]
        return {
            'hidden_dims': hidden_dims,
            'dropout': section.getfloat('dropout'),
            'use_batch_norm': section.getboolean('use_batch_norm')
        }
    
    def _load_model_lstm(self) -> Dict[str, Any]:
        """Load LSTM model configuration."""
        section = self.parser['MODEL_LSTM']
        return {
            'hidden_size': section.getint('hidden_size'),
            'num_layers': section.getint('num_layers'),
            'dropout': section.getfloat('dropout'),
            'fc_hidden_dim': section.getint('fc_hidden_dim'),
            'bidirectional': section.getboolean('bidirectional'),
            'sequence_length': section.getint('sequence_length')
        }
    
    def _load_model_bilstm(self) -> Dict[str, Any]:
        """Load BiLSTM model configuration."""
        section = self.parser['MODEL_BILSTM']
        return {
            'hidden_size': section.getint('hidden_size'),
            'num_layers': section.getint('num_layers'),
            'dropout': section.getfloat('dropout'),
            'fc_hidden_dim': section.getint('fc_hidden_dim'),
            'bidirectional': section.getboolean('bidirectional'),
            'sequence_length': section.getint('sequence_length')
        }
    
    def _load_model_cnn(self) -> Dict[str, Any]:
        """Load CNN model configuration."""
        section = self.parser['MODEL_CNN']
        return {
            'num_conv_layers': section.getint('num_conv_layers'),
            'base_channels': section.getint('base_channels'),
            'kernel_size': section.getint('kernel_size'),
            'dropout': section.getfloat('dropout'),
            'fc_hidden_dim': section.getint('fc_hidden_dim'),
            'sequence_length': section.getint('sequence_length')
        }
    
    def _load_model_cnn1d(self) -> Dict[str, Any]:
        """Load CNN1D model configuration."""
        section = self.parser['MODEL_CNN1D']
        return {
            'num_conv_layers': section.getint('num_conv_layers'),
            'base_channels': section.getint('base_channels'),
            'kernel_size': section.getint('kernel_size'),
            'dropout': section.getfloat('dropout'),
            'fc_hidden_dim': section.getint('fc_hidden_dim'),
            'sequence_length': section.getint('sequence_length')
        }
    
    def _load_model_hybrid(self) -> Dict[str, Any]:
        """Load Hybrid CNN-LSTM model configuration."""
        section = self.parser['MODEL_HYBRID']
        return {
            'cnn_channels': section.getint('cnn_channels'),
            'cnn_kernel_size': section.getint('cnn_kernel_size'),
            'lstm_hidden_size': section.getint('lstm_hidden_size'),
            'lstm_num_layers': section.getint('lstm_num_layers'),
            'dropout': section.getfloat('dropout'),
            'fc_hidden_dim': section.getint('fc_hidden_dim'),
            'sequence_length': section.getint('sequence_length')
        }
    
    def _load_model_transformer(self) -> Dict[str, Any]:
        """Load Transformer model configuration."""
        section = self.parser['MODEL_TRANSFORMER']
        return {
            'd_model': section.getint('d_model'),
            'nhead': section.getint('nhead'),
            'num_layers': section.getint('num_layers'),
            'dim_feedforward': section.getint('dim_feedforward'),
            'dropout': section.getfloat('dropout'),
            'fc_hidden_dim': section.getint('fc_hidden_dim'),
            'sequence_length': section.getint('sequence_length')
        }
    
    def _load_model_rf(self) -> Dict[str, Any]:
        """Load Random Forest model configuration."""
        section = self.parser['MODEL_RF']
        max_depth = section.get('max_depth')
        max_depth = int(max_depth) if max_depth.lower() != 'none' else None
        return {
            'n_estimators': section.getint('n_estimators'),
            'max_depth': max_depth,
            'min_samples_split': section.getint('min_samples_split'),
            'min_samples_leaf': section.getint('min_samples_leaf'),
            'max_features': section.get('max_features')
        }
    
    def _load_model_xgboost(self) -> Dict[str, Any]:
        """Load XGBoost model configuration."""
        section = self.parser['MODEL_XGBOOST']
        return {
            'n_estimators': section.getint('n_estimators'),
            'max_depth': section.getint('max_depth'),
            'learning_rate': section.getfloat('learning_rate'),
            'subsample': section.getfloat('subsample'),
            'colsample_bytree': section.getfloat('colsample_bytree'),
            'gamma': section.getfloat('gamma'),
            'reg_alpha': section.getfloat('reg_alpha'),
            'reg_lambda': section.getfloat('reg_lambda'),
            'min_child_weight': section.getint('min_child_weight')
        }
    
    def get_training_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get training configuration for specific model.
        
        Args:
            model_type: Model type (mlp, lstm, cnn, hybrid)
        
        Returns:
            Dictionary with n_epochs, batch_size, learning_rate, weight_decay, device
        """
        # For non-DL models, return basic config with defaults
        if model_type in ['rf', 'svm', 'logreg', 'xgboost', 'dt', 'knn']:
            return {
                'n_epochs': 0,
                'batch_size': 0,
                'learning_rate': 0.0,
                'weight_decay': 0.0,
                'test_size': self.data['test_size']
            }

        return {
            'n_epochs': self.training[f'epochs_{model_type}'],
            'batch_size': self.training[f'batch_size_{model_type}'],
            'learning_rate': self.training[f'learning_rate_{model_type}'],
            'weight_decay': self.training['weight_decay'],
            'early_stopping_patience': self.training['early_stopping_patience'],
            'pos_weight': self.training.get('pos_weight', None),
            'test_size': self.data['test_size']
        }
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get model architecture configuration.
        
        Args:
            model_type: Model type (mlp, lstm, cnn, hybrid, rf, xgboost)
        
        Returns:
            Dictionary with model-specific parameters
        """
        config_map = {
            'mlp': self.model_mlp,
            'lstm': self.model_lstm,
            'cnn': self.model_cnn,
            'hybrid': self.model_hybrid,
            'rf': self.model_rf,
            'xgboost': self.model_xgboost
        }
        return config_map.get(model_type, {})
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"\n[PATHS]")
        for key, val in self.paths.items():
            print(f"  {key}: {val}")
        
        print(f"\n[DATA]")
        for key, val in self.data.items():
            print(f"  {key}: {val}")
        
        print(f"\n[TRAINING]")
        for key, val in self.training.items():
            print(f"  {key}: {val}")
        
        print("=" * 70)


def load_config(config_file: str = "config.ini") -> Config:
    """
    Load configuration from file.
    
    Args:
        config_file: Path to config file
    
    Returns:
        Config object
    
    Example:
        >>> config = load_config()
        >>> print(config.data.time_window_s)
        8
    """
    return Config(config_file)


if __name__ == "__main__":
    config = load_config()
    config.print_summary()

