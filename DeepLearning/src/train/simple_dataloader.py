"""
Simple Data Loader for Filtered Eye-Tracking Data
=================================================
Loads data with configurable filters:
- Parts: Timer-Correct, Timer-No-Correct, No-Timer, etc.
- AOI: Answer_Area, Question, Timer, Submit, etc.
- Time Window: Last N seconds or full duration
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os


class EyeTrackingDataset:
    """
    Load and filter eye-tracking data with configurable parameters.
    
    Args:
        labels_csv: Path to stage3_with_labels.csv
        trials_dir: Directory containing trial CSV files
        parts_filter: List of parts to include (e.g., ['Timer-Correct', 'Timer-No-Correct'])
        aoi_filter: List of AOIs to include (e.g., ['Answer_Area'])
        time_window_s: Last N seconds to extract (None = full duration)
        feature_columns: List of feature columns to use (None = default set)
        sampling_rate: Eye-tracking sampling rate (default: 150 Hz)
        preprocessed_dir: Directory with pre-processed cleaned CSVs (optional)
    """
    
    def __init__(self, labels_csv=None, trials_dir=None, parts_filter=None, aoi_filter=None, 
                 time_window_s=None, feature_columns=None, sampling_rate=150, 
                 preprocessed_dir=None):
        self.labels_csv = labels_csv
        self.trials_dir = trials_dir
        self.parts_filter = parts_filter
        self.aoi_filter = aoi_filter
        self.time_window_s = time_window_s
        self.sampling_rate = sampling_rate
        self.preprocessed_dir = preprocessed_dir
        
        # Feature columns - use provided list or default
        if feature_columns is not None:
            self.feature_cols = feature_columns
        else:
            # If None, we will infer them from the data (ALL features)
            self.feature_cols = None
            
        # Metadata columns to exclude when inferring features
        self.metadata_cols = [
            'participant_id', 'question_id', 'part', 'is_valid', 'AOI', 
            'randomness_label', 'randomness_label_num', 'Unnamed: 0'
        ]

    def _infer_feature_columns(self, df):
        """Infer feature columns from dataframe by excluding metadata."""
        return [col for col in df.columns if col not in self.metadata_cols]
    
    def load_aggregated_features(self):
        """
        Load aggregated features (mean, std, min, max, nan_ratio).
        For MLP models.
        
        Returns:
            X: (n_samples, n_features) numpy array
            y: (n_samples,) numpy array
            sample_info: List of dicts with metadata
        """
        # If preprocessed_dir is provided, load directly from there
        if self.preprocessed_dir:
            return self._load_from_preprocessed()

        df_labels = pd.read_csv(self.labels_csv)
        
        # Filter out INVALID trials (they don't have corresponding CSV files)
        df_labels = df_labels[df_labels['randomness_label'] != 'INVALID'].copy()
        
        # Apply parts filter (if specified)
        if self.parts_filter:
            df_filtered = df_labels[df_labels['part'].isin(self.parts_filter)].copy()
        else:
            df_filtered = df_labels.copy()
            
        # Note: AOI filter will be applied when reading trial CSVs
        
        X_list = []
        y_list = []
        sample_info = []
        skipped = 0
        
        # Track skip reasons
        skip_reasons = {'file_not_found': 0, 'insufficient_data': 0, 'error': 0}
        
        for idx, row in df_filtered.iterrows():
            p_id = row['participant_id']
            q_id = row['question_id']
            filepath = os.path.join(self.trials_dir, f"{p_id}_{q_id}.csv")
            
            if not os.path.exists(filepath):
                skipped += 1
                skip_reasons['file_not_found'] += 1
                continue
            
            try:
                df_trial = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
                
                # INFER FEATURES CHECKS (Once per dataset load)
                if self.feature_cols is None:
                    self.feature_cols = self._infer_feature_columns(df_trial)
                    print(f"  [INFO] Inferred {len(self.feature_cols)} feature columns from first file.")
                    print(f"  [INFO] Features: {self.feature_cols[:5]} ...")
                
                # STEP 1: Extract time window FIRST (temporal slice before spatial filter)
                # This matches baseline behavior: tail(1200) THEN filter Answer Area
                if self.time_window_s is not None:
                    n_samples = int(self.time_window_s * self.sampling_rate)
                    df_tail = df_trial.tail(n_samples)
                else:
                    df_tail = df_trial
                
                # STEP 2: Filter AOI (spatial slice on the temporal window)
                if self.aoi_filter:
                    df_aoi = df_tail[df_tail['AOI'].isin(self.aoi_filter)]
                else:
                    df_aoi = df_tail  # Use all AOIs
                
                # STEP 3: Quality check - skip if too few samples in AOI during time window
                # Baseline requires at least 10 samples
                if len(df_aoi) < 10:
                    skipped += 1
                    skip_reasons['insufficient_data'] += 1
                    continue
                
                # Compute aggregated features
                features = []
                for col in self.feature_cols:
                    if col in df_aoi.columns:
                        values = df_aoi[col].values
                        features.extend([
                            np.nanmean(values),
                            np.nanstd(values),
                            np.nanmin(values),
                            np.nanmax(values),
                            np.isnan(values).mean()
                        ])
                    else:
                        features.extend([0, 0, 0, 0, 1])  # Missing column
                
                features = np.nan_to_num(features, nan=0.0)
                X_list.append(features)
                y_list.append(row['randomness_label_num'])  # Label column name
                
                sample_info.append({
                    'participant_id': p_id,
                    'question_id': q_id,
                    'part': row['part']
                })
                
            except Exception as e:
                print(f"Warning: Skipping {filepath}: {e}")
                skipped += 1
                skip_reasons['error'] += 1
                continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"[OK] Loaded {len(X)} samples ({skipped} skipped)")
        if skipped > 0:
            print(f"  Skip reasons: File not found={skip_reasons['file_not_found']}, Insufficient data (<10 samples in AOI)={skip_reasons['insufficient_data']}, Error={skip_reasons['error']}")
        if len(y) > 0:
            print(f"  Class 0: {(y==0).sum()} | Class 1: {(y==1).sum()}")
        print(f"  Features: {X.shape[1] if len(X) > 0 else 0}")
        
        return X, y, sample_info

    def _load_from_preprocessed(self):
        """
        Load features from a directory of pre-processed cleaned CSVs.
        """
        p_dir = Path(self.preprocessed_dir)
        files = list(p_dir.glob("*.csv"))
        print(f"Loading {len(files)} pre-processed trials from {p_dir}...")
        
        X_list = []
        y_list = []
        sample_info = []
        
        for f in files:
            try:
                df = pd.read_csv(f)
                
                # INFER FEATURES CHECKS (Once per dataset load)
                if self.feature_cols is None:
                    self.feature_cols = self._infer_feature_columns(df)
                    print(f"  [INFO] Inferred {len(self.feature_cols)} feature columns from first file.")

                # Assume label is valid since it's pre-processed
                label_str = df['randomness_label'].iloc[0]
                label = 1 if label_str == 'RANDOM' else 0
                
                # Check for insufficient data (though cleaned folder should be good)
                if len(df) < 5:
                    continue
                    
                features = []
                for col in self.feature_cols:
                    if col in df.columns:
                        series = df[col]
                        features.extend([
                            series.mean(),
                            series.std(),
                            series.min(),
                            series.max(),
                            series.isna().mean()
                        ])
                    else:
                        features.extend([0, 0, 0, 0, 1])
                
                X_list.append(features)
                y_list.append(label)
                sample_info.append({
                    'participant_id': f.name, # minimal info
                    'label': label_str
                })
                
            except Exception as e:
                print(f"Error loading {f.name}: {e}")
                
        return np.array(X_list), np.array(y_list), sample_info
    
    def load_sequences(self, max_length=100):
        """
        Load raw sequences for CNN/LSTM models.
        
        Args:
            max_length: Maximum sequence length (downsampled if longer)
        
        Returns:
            X: (n_samples, n_features, seq_length) numpy array
            y: (n_samples,) numpy array
            sample_info: List of dicts with metadata
        """
        df_labels = pd.read_csv(self.labels_csv)
        
        # Filter out INVALID trials (they don't have corresponding CSV files)
        df_labels = df_labels[df_labels['randomness_label'] != 'INVALID'].copy()
        
        # Apply parts filter (if specified)
        if self.parts_filter:
            df_filtered = df_labels[df_labels['part'].isin(self.parts_filter)].copy()
        else:
            df_filtered = df_labels.copy()
        
        # Note: AOI filter will be applied when reading trial CSVs
        
        sequences = []
        labels = []
        sample_info = []
        skipped = 0
        skip_reasons = {'file_not_found': 0, 'insufficient_data': 0, 'error': 0}
        
        for idx, row in df_filtered.iterrows():
            p_id = row['participant_id']
            q_id = row['question_id']
            filepath = os.path.join(self.trials_dir, f"{p_id}_{q_id}.csv")
            
            if not os.path.exists(filepath):
                skipped += 1
                skip_reasons['file_not_found'] += 1
                continue
            
            try:
                df_trial = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
                
                # INFER FEATURES CHECKS (Once per dataset load)
                if self.feature_cols is None:
                    self.feature_cols = self._infer_feature_columns(df_trial)
                    print(f"  [INFO] Inferred {len(self.feature_cols)} feature columns from first file.")
                
                # STEP 1: Extract time window FIRST (temporal slice before spatial filter)
                if self.time_window_s is not None:
                    n_samples = int(self.time_window_s * self.sampling_rate)
                    df_tail = df_trial.tail(n_samples)
                else:
                    df_tail = df_trial
                
                # STEP 2: Filter AOI (spatial slice on the temporal window)
                if self.aoi_filter:
                    df_aoi = df_tail[df_tail['AOI'].isin(self.aoi_filter)]
                else:
                    df_aoi = df_tail  # Use all AOIs
                
                # STEP 3: Quality check - skip if too few samples
                if len(df_aoi) < 10:
                    skipped += 1
                    skip_reasons['insufficient_data'] += 1
                    continue
                
                # Extract sequence
                sequence = df_aoi[self.feature_cols].values
                sequence = np.nan_to_num(sequence, nan=0.0)
                
                # Downsample if needed
                if len(sequence) > max_length:
                    step = len(sequence) // max_length
                    sequence = sequence[::step][:max_length]
                
                # Pad if needed
                if len(sequence) < max_length:
                    padding = np.zeros((max_length - len(sequence), len(self.feature_cols)))
                    sequence = np.vstack([sequence, padding])
                
                sequences.append(sequence)
                labels.append(row['randomness_label_num'])  # Label column name
                
                sample_info.append({
                    'participant_id': p_id,
                    'question_id': q_id,
                    'part': row['part']
                })
                
            except Exception as e:
                print(f"Warning: Skipping {filepath}: {e}")
                skipped += 1
                skip_reasons['error'] += 1
                continue
        
        # Convert to numpy arrays
        all_sequences = np.array(sequences)  # (n_samples, seq_len, n_features)
        all_labels = np.array(labels)
        
        # Transpose to (n_samples, n_features, seq_len) for Conv1D
        all_sequences = all_sequences.transpose(0, 2, 1)
        
        print(f"[OK] Loaded {len(all_sequences)} sequences ({skipped} skipped)")
        if skipped > 0:
            print(f"  Skip reasons: File not found={skip_reasons['file_not_found']}, Insufficient data (<10 samples in AOI)={skip_reasons['insufficient_data']}, Error={skip_reasons['error']}")
        print(f"  Class 0: {(all_labels==0).sum()} | Class 1: {(all_labels==1).sum()}")
        print(f"  Shape: {all_sequences.shape}")
        
        return all_sequences, all_labels, sample_info


def prepare_dataloaders(X, y, test_size=0.2, batch_size=8, sequence_data=False):
    """
    Prepare train/test data loaders.
    
    Args:
        X: Features or sequences
        y: Labels
        test_size: Test split ratio
        batch_size: Batch size
        sequence_data: If False, normalize features. If True, normalize sequences.
    
    Returns:
        train_loader, test_loader, scaler, y_train, y_test
    """
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Normalize
    if not sequence_data:
        # For MLP: (samples, features)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        # For CNN/LSTM: (samples, features, seq_len)
        scaler = StandardScaler()
        n_samples, n_features, seq_len = X_train.shape
        
        # Reshape to (samples * seq_len, features)
        X_train_flat = X_train.transpose(0, 2, 1).reshape(-1, n_features)
        X_test_flat = X_test.transpose(0, 2, 1).reshape(-1, n_features)
        
        # Normalize
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_test_flat = scaler.transform(X_test_flat)
        
        # Reshape back
        X_train = X_train_flat.reshape(n_samples, seq_len, n_features).transpose(0, 2, 1)
        X_test = X_test_flat.reshape(len(X_test), seq_len, n_features).transpose(0, 2, 1)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"[OK] Data loaders ready")
    print(f"  Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    
    return train_loader, test_loader, scaler, y_train, y_test
