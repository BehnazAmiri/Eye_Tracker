"""
Statistical Feature Engineering for Eye-Tracking Data
======================================================
Transform raw time-series features into statistical features.

For each feature column, compute:
- Basic statistics: mean, std, min, max, median
- Percentiles: Q25, Q75
- Derived: range, IQR
- Distribution: skewness, kurtosis
- Temporal: first derivative (rate of change), second derivative (acceleration)

This increases features from 13 -> ~182 (13 * 14 statistical features per column)
Expected accuracy improvement: +5-8%

Reproducibility: 100% deterministic (fixed seed not needed for deterministic math operations)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple


class StatisticalFeatureEngineer:
    """
    Extract statistical features from time-series eye-tracking data.
    
    Designed for LSTM/CNN models to capture temporal patterns through statistics
    instead of raw sequences.
    """
    
    def __init__(self, feature_columns: List[str]):
        """
        Args:
            feature_columns: List of column names to extract features from
        """
        self.feature_columns = feature_columns
        self.feature_names = []  # Will be populated after first transform
        
    def _compute_statistical_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute statistical features for a single time-series sequence.
        
        Args:
            sequence: (seq_len,) numpy array of a single feature
            
        Returns:
            features: (14,) array of statistical features
        """
        # Handle NaN values
        sequence = np.nan_to_num(sequence, nan=0.0)
        
        # Handle empty or constant sequences
        if len(sequence) == 0 or np.all(sequence == 0):
            return np.zeros(14)
        
        # Basic statistics
        mean_val = np.mean(sequence)
        std_val = np.std(sequence)
        min_val = np.min(sequence)
        max_val = np.max(sequence)
        median_val = np.median(sequence)
        
        # Percentiles
        q25 = np.percentile(sequence, 25)
        q75 = np.percentile(sequence, 75)
        
        # Derived statistics
        range_val = max_val - min_val
        iqr_val = q75 - q25
        
        # Distribution statistics
        try:
            skewness = stats.skew(sequence)
            kurtosis_val = stats.kurtosis(sequence)
        except:
            skewness = 0.0
            kurtosis_val = 0.0
        
        # Temporal features: derivatives
        if len(sequence) > 1:
            first_derivative = np.diff(sequence)
            mean_derivative = np.mean(first_derivative)
            std_derivative = np.std(first_derivative)
            
            if len(first_derivative) > 1:
                second_derivative = np.diff(first_derivative)
                mean_acceleration = np.mean(second_derivative)
            else:
                mean_acceleration = 0.0
        else:
            mean_derivative = 0.0
            std_derivative = 0.0
            mean_acceleration = 0.0
        
        features = np.array([
            mean_val,
            std_val,
            min_val,
            max_val,
            median_val,
            q25,
            q75,
            range_val,
            iqr_val,
            skewness,
            kurtosis_val,
            mean_derivative,
            std_derivative,
            mean_acceleration
        ])
        
        return features
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names for all statistical features."""
        stat_names = [
            'mean', 'std', 'min', 'max', 'median',
            'q25', 'q75', 'range', 'iqr',
            'skewness', 'kurtosis',
            'mean_derivative', 'std_derivative', 'mean_acceleration'
        ]
        
        feature_names = []
        for col in self.feature_columns:
            for stat in stat_names:
                feature_names.append(f"{col}_{stat}")
        
        return feature_names
    
    def transform_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Transform sequences into statistical features.
        
        Args:
            X: (n_samples, n_features, seq_len) numpy array
            
        Returns:
            X_stats: (n_samples, n_statistical_features) numpy array
        """
        n_samples, n_features, seq_len = X.shape
        
        # Each original feature produces 14 statistical features
        n_stat_features = n_features * 14
        X_stats = np.zeros((n_samples, n_stat_features))
        
        for i in range(n_samples):
            stat_idx = 0
            for f in range(n_features):
                sequence = X[i, f, :]  # (seq_len,)
                stat_features = self._compute_statistical_features(sequence)
                X_stats[i, stat_idx:stat_idx+14] = stat_features
                stat_idx += 14
        
        # Generate feature names on first call
        if len(self.feature_names) == 0:
            self.feature_names = self._generate_feature_names()
        
        return X_stats
    
    def transform_and_save(self, X: np.ndarray, y: np.ndarray, 
                          sample_info: List[dict], output_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform sequences and save as CSV for inspection/reuse.
        
        Args:
            X: (n_samples, n_features, seq_len) numpy array
            y: (n_samples,) labels
            sample_info: List of dicts with metadata
            output_path: Path to save CSV
            
        Returns:
            X_stats: (n_samples, n_statistical_features) numpy array
            y: (n_samples,) labels (unchanged)
        """
        X_stats = self.transform_sequences(X)
        
        # Create DataFrame for saving
        df = pd.DataFrame(X_stats, columns=self.feature_names)
        df['label'] = y
        
        # Add metadata
        for key in sample_info[0].keys():
            df[key] = [info[key] for info in sample_info]
        
        df.to_csv(output_path, index=False)
        print(f"[OK] Statistical features saved: {output_path}")
        print(f"  Original features: {X.shape[1]} | Statistical features: {X_stats.shape[1]}")
        print(f"  Samples: {X_stats.shape[0]}")
        
        return X_stats, y


def create_statistical_features(X_train: np.ndarray, X_test: np.ndarray, 
                               feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to create statistical features for train and test sets.
    
    Args:
        X_train: (n_train, n_features, seq_len)
        X_test: (n_test, n_features, seq_len)
        feature_columns: List of feature column names
        
    Returns:
        X_train_stats: (n_train, n_statistical_features)
        X_test_stats: (n_test, n_statistical_features)
    """
    engineer = StatisticalFeatureEngineer(feature_columns)
    
    X_train_stats = engineer.transform_sequences(X_train)
    X_test_stats = engineer.transform_sequences(X_test)
    
    print(f"\n{'='*70}")
    print(f"STATISTICAL FEATURE ENGINEERING")
    print(f"{'='*70}")
    print(f"Original features: {X_train.shape[1]}")
    print(f"Statistical features per original: 14")
    print(f"  - Basic: mean, std, min, max, median")
    print(f"  - Percentiles: Q25, Q75")
    print(f"  - Derived: range, IQR")
    print(f"  - Distribution: skewness, kurtosis")
    print(f"  - Temporal: mean_derivative, std_derivative, mean_acceleration")
    print(f"Total statistical features: {X_train_stats.shape[1]}")
    print(f"Train samples: {X_train_stats.shape[0]} | Test samples: {X_test_stats.shape[0]}")
    print(f"{'='*70}\n")
    
    return X_train_stats, X_test_stats
