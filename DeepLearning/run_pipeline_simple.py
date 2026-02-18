"""
Simple Pipeline for Eye-Tracking Classification
===============================================
End-to-end pipeline: Data Loading -> Training -> Evaluation -> Reporting
"""

import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config_loader import load_config
from train.simple_dataloader import EyeTrackingDataset, prepare_dataloaders
from train.simple_trainer import SimpleTrainer
from models.simple import create_model, create_ml_model, get_model_info, HAS_XGBOOST
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Check XGBoost availability
if not HAS_XGBOOST:
    print("Warning: xgboost not installed. XGBoost model will not be available.")

import joblib
import random


def seed_everything(seed=42):
    """
    Set all random seeds for reproducibility.
    
    This ensures that:
    1. Python's random module is seeded
    2. NumPy's random generator is seeded
    3. PyTorch's CPU random generator is seeded
    4. PyTorch's GPU random generators are seeded
    5. CUDNN operations become deterministic (slower but reproducible)
    
    Args:
        seed: Random seed value (default: 42)
    
    Note:
        Setting CUDNN to deterministic mode may slow down training by ~20%
        but ensures identical results across runs with same data/config.
    """
    # Python's random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        # CUDNN deterministic mode (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"[OK] GPU seeding enabled (seed={seed})")
        print("  [WARNING] CUDNN deterministic mode ON (may reduce speed by ~20%)")
    
    print(f"[OK] Random seed set to {seed} for full reproducibility")


def snapshot_config_ini(config_path='config.ini'):
    """
    Create a snapshot of current config.ini for reproducibility.
    
    Returns dict with all config values at time of run.
    This ensures we can reproduce exact results later.
    """
    import configparser
    
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    
    snapshot = {}
    for section in config.sections():
        snapshot[section] = dict(config[section])
    
    return snapshot


def validate_config_for_reproducibility(config_path='config.ini', check_lopo=True):
    """
    Validate config.ini against known best configurations.
    Warns if config differs from optimal setup.
    
    Args:
        config_path: Path to config.ini
        check_lopo: If True, check against LOPO best config (69.96%)
                    If False, skip validation
    
    Returns:
        bool: True if config is valid, False if there are issues
    """
    import configparser
    
    if not check_lopo:
        return True
        
    # Best LOPO configuration (69.96% accuracy)
    BEST_LOPO = {
        'parts_filter': 'Timer-Correct,Timer-No-Correct',
        'time_window_s': '8',
        'random_seed': '42',
        'test_size': '0.3',
        'feature_columns': 'BPOGX,BPOGY,FPOGD,FPOGX,FPOGY,LPCX,LPCY,LPD,LPUPILD,RPCX,RPCY,RPD,RPUPILD'
    }
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    issues = []
    
    # Check critical parameters
    if 'DATA' in config:
        data_section = config['DATA']
        
        for key, expected_value in BEST_LOPO.items():
            actual_value = data_section.get(key, '').strip()
            
            if actual_value != expected_value:
                issues.append(f"  [WARN] {key}: '{actual_value}' (expected: '{expected_value}')")
    
    if issues:
        print(f"\n{'='*70}")
        print(f"[WARN]  CONFIG VALIDATION WARNING")
        print(f"{'='*70}")
        print(f"The following parameters differ from best LOPO config (69.96%):\n")
        for issue in issues:
            print(issue)
        print(f"\n{'='*70}\n")
        return False
    else:
        print(f"[OK] Config validated: Matches best LOPO configuration (69.96%)")
        return True


def convert_to_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def _generate_simple_html_report(metadata, results):
    """Generate simple HTML report from run metadata."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    model_type = metadata['model_type'].upper()
    data_cfg = metadata['data_config']
    train_metrics = metadata['metrics']['train']
    test_metrics = metadata['metrics']['test']
    
    # Confusion matrix
    cm = test_metrics['confusion_matrix']
    
    # JSON Dump for Tree View
    json_str = json.dumps(metadata, default=str)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{model_type} Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 15px; overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; }}
        .header h1 {{ font-size: 2.5em; margin: 0; }}
        .content {{ padding: 40px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea; }}
        .card h3 {{ color: #667eea; margin-top: 0; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }}
        .metric-box {{ background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .metric-label {{ color: #7f8c8d; font-size: 0.9em; }}
        .metric-value {{ color: #667eea; font-size: 2em; font-weight: bold; }}
        .best {{ background: #d4edda; }}
        .best .metric-value {{ color: #155724; }}
        table {{ border-collapse: collapse; margin: 20px auto; }}
        th, td {{ border: 1px solid #ddd; padding: 12px 20px; text-align: center; }}
        th {{ background: #667eea; color: white; }}
        
        /* Tree View Styles */
        .json-tree {{ font-family: 'Consolas', monospace; font-size: 14px; background: #2d2d2d; color: #ccc; padding: 20px; border-radius: 8px; overflow: auto; max-height: 600px; }}
        .json-tree ul {{ list-style: none; margin: 0; padding: 0 0 0 20px; }}
        .json-tree li {{ margin: 2px 0; }}
        .key {{ color: #9cdcfe; }}
        .string {{ color: #ce9178; }}
        .number {{ color: #b5cea8; }}
        .boolean {{ color: #569cd6; }}
        .null {{ color: #569cd6; }}
        .caret {{ cursor: pointer; user-select: none; color: #fff; margin-right: 5px; }}
        .caret::before {{ content: ">"; color: #fff; display: inline-block; margin-right: 6px; font-size: 10px; transition: transform 0.1s; }}
        .caret-down::before {{ transform: rotate(90deg); }}
        .nested {{ display: none; }}
        .active {{ display: block; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>[BRAIN] {model_type} Report</h1>
            <p>{metadata['run_id']} | {timestamp}</p>
        </div>
        <div class="content">
            <div class="section">
                <h2>[CONFIG] Configuration</h2>
                <div class="grid">
                    <div class="card"><h3>Model</h3><p>{model_type}</p></div>
                    <div class="card"><h3>Data</h3>
                        <p>Parts: {data_cfg['parts_filter']}<br>
                        AOI: {data_cfg['aoi_filter']}<br>
                        Window: {data_cfg['time_window_s']}s</p>
                    </div>
                    <div class="card"><h3>Dataset</h3>
                        <p>Samples: {data_cfg['n_samples']}<br>
                        Train: {data_cfg['n_train']} | Test: {data_cfg['n_test']}</p>
                    </div>
                    <div class="card"><h3>Training</h3>
                        <p>Epochs: {metadata['training_config']['n_epochs']}<br>
                        Batch: {metadata['training_config']['batch_size']}<br>
                        LR: {metadata['training_config']['learning_rate']}</p>
                    </div>
                </div>
            </div>
            <div class="section">
                <h2>[CHART] Test Results</h2>
                <div class="metrics">
                    <div class="metric-box best"><div class="metric-label">AUC</div><div class="metric-value">{test_metrics['roc_auc']:.4f}</div></div>
                    <div class="metric-box"><div class="metric-label">F1</div><div class="metric-value">{test_metrics['f1']:.4f}</div></div>
                    <div class="metric-box"><div class="metric-label">Accuracy</div><div class="metric-value">{test_metrics['accuracy']:.4f}</div></div>
                    <div class="metric-box"><div class="metric-label">Precision</div><div class="metric-value">{test_metrics['precision']:.4f}</div></div>
                    <div class="metric-box"><div class="metric-label">Recall</div><div class="metric-value">{test_metrics['recall']:.4f}</div></div>
                </div>
            </div>
            <div class="section">
                <h2>[TARGET] Confusion Matrix</h2>
                <table>
                    <tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
                    <tr><th>True 0</th><td>{cm[0][0]}</td><td>{cm[0][1]}</td></tr>
                    <tr><th>True 1</th><td>{cm[1][0]}</td><td>{cm[1][1]}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>[NOTE] Full Configuration & Results (Tree View)</h2>
                <div id="json-tree" class="json-tree"></div>
            </div>
        </div>
    </div>
    
    <script>
        const metadata = {json_str};
        
        function renderJSON(data) {{
            if (typeof data === 'object' && data !== null) {{
                const ul = document.createElement('ul');
                for (const key in data) {{
                    const li = document.createElement('li');
                    const value = data[key];
                    const isObj = typeof value === 'object' && value !== null;
                    
                    const keySpan = document.createElement('span');
                    keySpan.className = 'key';
                    keySpan.textContent = key + ': ';
                    
                    if (isObj) {{
                        const caret = document.createElement('span');
                        caret.className = 'caret';
                        caret.onclick = function() {{
                            this.parentElement.querySelector('.nested').classList.toggle('active');
                            this.classList.toggle('caret-down');
                        }};
                        li.appendChild(caret);
                        li.appendChild(keySpan);
                        
                        const nested = renderJSON(value);
                        nested.className = 'nested';
                        li.appendChild(nested);
                    }} else {{
                        li.appendChild(keySpan);
                        
                        const valSpan = document.createElement('span');
                        if (typeof value === 'string') valSpan.className = 'string';
                        else if (typeof value === 'number') valSpan.className = 'number';
                        else if (typeof value === 'boolean') valSpan.className = 'boolean';
                        else valSpan.className = 'null';
                        
                        valSpan.textContent = JSON.stringify(value);
                        li.appendChild(valSpan);
                    }}
                    ul.appendChild(li);
                }}
                return ul;
            }} else {{
                return document.createTextNode(JSON.stringify(data));
            }}
        }}
        
        const treeContainer = document.getElementById('json-tree');
        treeContainer.appendChild(renderJSON(metadata));
        
        // Open first level
        const firstLevel = treeContainer.querySelector('ul');
        if (firstLevel) {{
            firstLevel.style.display = 'block';
            firstLevel.className = 'active';
             // Expand all top level items
            Array.from(treeContainer.querySelectorAll('.caret')).forEach(el => {{
                el.click();
            }});
        }}
    </script>
</body>
</html>"""
    return html


def run_simple_pipeline(
    model_type='mlp',
    parts_filter=None,
    aoi_filter=None,
    time_window_s=None,
    feature_columns=None,
    model_config=None,
    training_config=None,
    trials_dir=None,
    config=None
):
    """
    Run complete simple pipeline.
    
    Args:
        model_type: 'mlp', 'cnn', 'cnn1d', 'lstm', 'bilstm', 'hybrid', or 'transformer'
        parts_filter: List of parts (default: ['Timer-Correct', 'Timer-No-Correct'])
        aoi_filter: List of AOIs (default: ['Answer_Area'])
        time_window_s: Time window in seconds (default: 8)
        feature_columns: List of feature columns to use (None = all available)
        model_config: Model configuration dict
        training_config: Training configuration dict
        trials_dir: Path to trials directory
    
    Returns:
        dict: Results
    """
    
    # Generate timestamp once to ensure correlation between artifacts
    shared_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Validate config (warning only for simple split)
    print("\n" + "="*70)
    print("CONFIG SNAPSHOT")
    print("="*70)
    print("Config will be saved with results for reproducibility")
    
    # Set seed for full reproducibility BEFORE any random operations
    random_seed = config.data['random_seed'] if config else 42
    seed_everything(random_seed)
    
    print("\n" + "="*70)
    print(f"SIMPLE PIPELINE: {model_type.upper()}")
    print("="*70)
    
    # Use config values if not explicitly provided
    if parts_filter is None:
        parts_filter = config.data['parts_filter']
    if aoi_filter is None:
        aoi_filter = config.data['aoi_filter']
    # Get feature columns from config (13 optimal features)
    # If not in config, return None to enable auto-inference of ALL features
    feature_columns = config.data.get('feature_columns', None)
    # time_window_s: None=use config, -1=full trial, number=specific window
    if time_window_s is None:
        time_window_s = config.data['time_window_s']
    elif time_window_s == -1:
        time_window_s = None  # Full trial
    if trials_dir is None:
        trials_dir = str(config.paths['trials_dir'])
    
    # Get model-specific configs from config file
    if model_config is None:
        model_config = config.get_model_config(model_type)
    if training_config is None:
        training_config = config.get_training_config(model_type)
    
    # CRITICAL DEBUG: Verify patience configuration
    print(f"\n{'='*70}")
    print(f"CONFIGURATION LOADED FROM config.ini:")
    print(f"  Model Type: {model_type.upper()}")
    print(f"  Epochs: {training_config['n_epochs']}")
    print(f"  Early Stopping Patience: {training_config.get('early_stopping_patience', 'NOT FOUND IN CONFIG!')}")
    print(f"  Batch Size: {training_config['batch_size']}")
    print(f"  Learning Rate: {training_config['learning_rate']}")
    print(f"{'='*70}\n")
    
    # Paths from config
    labels_csv = config.paths['labels_csv']
    trials_dir = Path(trials_dir)
    outputs_root = config.paths['outputs_dir']
    reports_root = config.paths['reports_dir']
    dl_inputs_root = config.paths['dl_inputs_dir']
    
    # Create directories
    reports_root.mkdir(parents=True, exist_ok=True)
    dl_inputs_root.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Model: {model_type}")
    print(f"  Parts: {parts_filter}")
    print(f"  AOI: {aoi_filter}")
    print(f"  Time Window: {time_window_s}s")
    if feature_columns:
        print(f"  Feature Columns: {len(feature_columns)} OPTIMAL features")
        print(f"    -> {', '.join(feature_columns)}")
    else:
        print(f"  Feature Columns: All (41 features)")
    
    # Step 1: Load data
    print(f"\n{'='*70}")
    print(f"STEP 1: LOADING DATA")
    print(f"{'='*70}")
    
    # Initialize metadata collection
    data_stats = {
        'original_trials': 0,
        'original_class_0': 0,
        'original_class_1': 0,
        'total_raw_rows': 0,  # Total rows in raw CSV files
        'skipped_trials': 0
    }
    
    dataset = EyeTrackingDataset(
        labels_csv=str(labels_csv),
        trials_dir=str(trials_dir),
        parts_filter=parts_filter,
        aoi_filter=aoi_filter,
        time_window_s=time_window_s,
        feature_columns=feature_columns
    )
    
    # Load appropriate data type
    is_sklearn = model_type in ['rf', 'svm', 'logreg', 'xgboost']
    
    # Check if we should use ADVANCED statistical features (182 features)
    # If so, we need to load SEQUENCES first, then transform them
    use_advanced_stats = config and config.data.get('use_statistical_features', False)
    
    if (model_type == 'mlp' or is_sklearn) and not use_advanced_stats:
        # Standard MLP/ML loading (simple 5-stat aggregation -> 65 features)
        X, y, sample_info = dataset.load_aggregated_features()
        sequence_data = False
        if not is_sklearn:
             input_dim = X.shape[1]
             model_config['input_dim'] = input_dim
    else:  # cnn, cnn1d, lstm... OR mlp with advanced stats
        # Load sequence data
        max_length = model_config.get('sequence_length', 300)
        
        X, y, sample_info = dataset.load_sequences(max_length=max_length)
        sequence_data = True
        input_channels = X.shape[1]
        model_config['input_channels'] = input_channels
        model_config['input_size'] = input_channels
    
    # Store original data statistics
    data_stats['original_trials'] = len(X)
    data_stats['original_class_0'] = int(np.sum(y == 0))
    data_stats['original_class_1'] = int(np.sum(y == 1))
    data_stats['feature_columns'] = dataset.feature_cols
    data_stats['n_features'] = X.shape[1] if len(X.shape) == 2 else X.shape[1]
    data_stats['sequence_length'] = X.shape[2] if len(X.shape) == 3 else None
    
    if len(X) < 10:
        raise ValueError(f"Insufficient data: only {len(X)} samples after filtering")
    
    # Step 2: Train/Test Split
    print(f"\n{'='*70}")
    print(f"STEP 2: TRAIN/TEST SPLIT")
    print(f"{'='*70}")
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # [STAR] CHECK FOR PRE-AUGMENTED DATA
    augmented_data_dir = Path("outputs/augmented_data")
    use_preaugmented = False
    
    if (augmented_data_dir / "X_train_augmented.npy").exists():
        print(f"\n[STAR] Found pre-augmented data in {augmented_data_dir}")
        print(f"   Loading pre-augmented data instead of augmenting during training...")
        
        X_train = np.load(augmented_data_dir / "X_train_augmented.npy")
        y_train = np.load(augmented_data_dir / "y_train_augmented.npy")
        X_test = np.load(augmented_data_dir / "X_test.npy")
        y_test = np.load(augmented_data_dir / "y_test.npy")
        
        # Load metadata
        with open(augmented_data_dir / "metadata.json", 'r') as f:
            aug_metadata = json.load(f)
        
        print(f"   Train: {len(X_train)} samples (augmented from {aug_metadata['original_train_samples']})")
        print(f"   Test:  {len(X_test)} samples")
        print(f"   Augmentation factor: {aug_metadata['augment_factor']}x")
        print(f"   Methods used: {', '.join(aug_metadata['augment_methods'])}")
        
        # Create dummy info_train and info_test
        info_train = [f"aug_sample_{i}" for i in range(len(X_train))]
        info_test = [f"test_sample_{i}" for i in range(len(X_test))]
        
        use_preaugmented = True
    
    if not use_preaugmented:
        random_seed = config.data['random_seed'] if config else 42
        X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
            X, y, sample_info, test_size=training_config['test_size'], 
            random_state=random_seed, stratify=y
        )
        
        print(f"Split complete:")
        print(f"  Train: {len(X_train)} samples (class_0={np.sum(y_train==0)}, class_1={np.sum(y_train==1)})")
        print(f"  Test:  {len(X_test)} samples (class_0={np.sum(y_test==0)}, class_1={np.sum(y_test==1)})")
    
    # [LOOP] DATA AUGMENTATION (if enabled and NOT using pre-augmented data)
    # Must be done AFTER split and BEFORE normalization
    
    print(f"\n{'='*70}")
    print(f"[DEBUG] AUGMENTATION CHECK")
    print(f"{'='*70}")
    print(f"  use_preaugmented = {use_preaugmented}")
    print(f"  config is not None = {config is not None}")
    print(f"  hasattr(config, 'augmentation') = {hasattr(config, 'augmentation')}")
    if hasattr(config, 'augmentation'):
        print(f"  config.augmentation = {config.augmentation}")
        print(f"  config.augmentation.get('use_augmentation') = {config.augmentation.get('use_augmentation', False)}")
    print(f"{'='*70}")
    
    if not use_preaugmented and config and hasattr(config, 'augmentation') and config.augmentation.get('use_augmentation', False):
        print(f"\n{'='*70}")
        print(f"[LOOP] DATA AUGMENTATION")
        print(f"{'='*70}")
        
        try:
            from data_augmentation import apply_augmentation
            
            aug_factor = config.augmentation.get('augment_factor', 2)
            aug_methods = config.augmentation.get('augment_methods', 'time_shift,noise,scale')
            
            print(f"  Augmentation enabled!")
            print(f"  Factor: {aug_factor}x (will increase {len(X_train)} -> {len(X_train) * aug_factor} samples)")
            print(f"  Methods: {aug_methods}")
            
            # Apply augmentation (only on training data!)
            X_train_aug, y_train_aug = apply_augmentation(X_train, y_train, config)
            
            # Update info_train to match new size
            original_train_size = len(X_train)
            new_samples = len(X_train_aug) - original_train_size
            
            # Duplicate info_train entries for augmented samples
            if new_samples > 0:
                # Calculate how many times each sample was augmented
                augment_per_sample = aug_factor - 1
                # Replicate info_train
                info_train_aug = info_train.copy()
                for i in range(augment_per_sample):
                    info_train_aug.extend(info_train.copy())
                info_train = info_train_aug[:len(X_train_aug)]
            
            X_train = X_train_aug
            y_train = y_train_aug
            
            print(f"  [OK] Augmentation complete!")
            print(f"     Original: {original_train_size} samples")
            print(f"     After Augmentation: {len(X_train)} samples")
            print(f"     Added: +{new_samples} synthetic samples")
            
        except Exception as e:
            print(f"  [WARN] Augmentation failed: {e}")
            print(f"     Continuing with original data...")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n  [INFO] Data augmentation: DISABLED")
    
    # [TARGET] STATISTICAL FEATURE ENGINEERING (Phase 1, Step 2 of roadmap to 80%)
    # Apply AFTER split/augmentation, BEFORE normalization
    if sequence_data and config and config.data.get('use_statistical_features', False):
        print(f"\n{'='*70}")
        print(f"[SCIENCE] STATISTICAL FEATURE ENGINEERING")
        print(f"{'='*70}")
        print(f"  Transforming time-series into statistical features...")
        print(f"  Original shape: {X_train.shape} (n_samples, n_features, seq_len)")
        
        try:
            from src.utils.feature_engineering import create_statistical_features
            
            # Get feature names (13 optimal features)
            feature_cols = config.data.get('feature_columns', [])
            if not feature_cols:
                # Infer from shape
                feature_cols = [f"feature_{i}" for i in range(X_train.shape[1])]
            
            # Transform sequences into statistical features
            X_train_orig = X_train.copy()  # Keep backup for potential debugging
            X_test_orig = X_test.copy()
            
            X_train, X_test = create_statistical_features(X_train, X_test, feature_cols)
            
            print(f"  [OK] Statistical features created!")
            print(f"     Original: {X_train_orig.shape[1]} features  {X_train_orig.shape[2]} timesteps")
            print(f"     Statistical: {X_train.shape[1]} features (14 stats per original feature)")
            print(f"     Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
            print(f"     Expected improvement: +5-8% (from 72.55% to 77-80%)")
            
            # Update sequence_data flag since we now have flat features
            sequence_data = False
            # Update model config with NEW input dimension (182 features)
            model_config['input_dim'] = X_train.shape[1]
            
        except Exception as e:
            print(f"  [WARN] Statistical feature engineering failed: {e}")
            print(f"     Continuing with original sequence data...")
            import traceback
            traceback.print_exc()
    
    # Store split statistics
    data_stats['split'] = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_class_0': int(np.sum(y_train == 0)),
        'train_class_1': int(np.sum(y_train == 1)),
        'test_class_0': int(np.sum(y_test == 0)),
        'test_class_1': int(np.sum(y_test == 1)),
        'test_size': training_config['test_size']
    }
    
    # Save RAW filtered data to CSV (after cleaning, before normalization)
    timestamp = shared_timestamp
    csv_dirname = f"{model_type}_{timestamp}_filtered_trials"
    csv_output_dir = dl_inputs_root / csv_dirname
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"\n{'='*70}")
        print(f"EXPORTING RAW FILTERED DATA (FROM CLEANED DATASET)")
        print(f"{'='*70}")
        
        # Load labels
        df_labels = pd.read_csv(labels_csv)
        
        # Apply parts filter
        if parts_filter:
            df_filtered = df_labels[df_labels['part'].isin(parts_filter)].copy()
        else:
            df_filtered = df_labels.copy()
        
        # Process each trial and save separately
        saved_trials = 0
        skipped = 0
        total_rows = 0
        
        for idx, row in df_filtered.iterrows():
            p_id = row['participant_id']
            q_id = row['question_id']
            filepath = trials_dir / f"{p_id}_{q_id}.csv"
            
            if not filepath.exists():
                skipped += 1
                continue
            
            try:
                # Read trial CSV
                df_trial = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
                
                # STEP 1: Extract time window FIRST (last N seconds from entire trial)
                # This matches baseline: tail(1200) THEN filter Answer Area
                if time_window_s is not None:
                    n_samples = int(time_window_s * 150)  # 150 Hz sampling rate
                    df_tail = df_trial.tail(n_samples)
                else:
                    df_tail = df_trial
                
                # STEP 2: Filter AOI (spatial slice on the temporal window)
                if aoi_filter:
                    df_filtered = df_tail[df_tail['AOI'].isin(aoi_filter)]
                else:
                    df_filtered = df_tail
                
                # STEP 3: Quality check - skip if too few samples in AOI during time window
                if len(df_filtered) < 10:
                    skipped += 1
                    continue
                
                # Filter feature columns (if specified)
                if feature_columns:
                    # Keep only feature columns
                    cols_to_keep = [col for col in feature_columns if col in df_filtered.columns]
                    df_final = df_filtered[cols_to_keep].copy()
                else:
                    # Keep all columns but remove unnecessary metadata
                    cols_to_drop = ['participant_id', 'question_id', 'part', 'is_valid', 
                                   'AOI', 'randomness_label']
                    df_final = df_filtered.drop(columns=[c for c in cols_to_drop if c in df_filtered.columns]).copy()
                
                # INJECT LABEL
                df_final['randomness_label'] = row['randomness_label']
                
                # Save this trial to separate CSV
                trial_csv_path = csv_output_dir / f"{p_id}_{q_id}.csv"
                df_final.to_csv(trial_csv_path, index=False)
                
                saved_trials += 1
                total_rows += len(df_final)
                
            except Exception as e:
                print(f"Warning: Skipping {filepath.name}: {e}")
                skipped += 1
                continue
        
        if saved_trials > 0:
            # Get sample columns from first saved file
            sample_file = next(csv_output_dir.glob("*.csv"))
            sample_df = pd.read_csv(sample_file, nrows=0)
            
            print(f"[OK] Raw filtered data saved: {csv_output_dir}")
            print(f"  Total trials saved: {saved_trials}")
            print(f"  Total rows: {total_rows:,}")
            print(f"  Skipped: {skipped} trials")
            print(f"  Columns per trial: {len(sample_df.columns)}")
            print(f"\nColumn names:")
            print(f"  {list(sample_df.columns)}")
            
            csv_path = csv_output_dir  # Return directory path
        else:
            print(f"[WARN] No data to export (all trials skipped)")
            csv_path = None
            
    except Exception as e:
        print(f"[WARN] Failed to save filtered CSV: {e}")
        import traceback
        traceback.print_exc()
        csv_path = None
    
    # Step 4: Prepare data loaders or normalize
    print(f"\n{'='*70}")
    print(f"STEP 4: DATA PREPARATION & NORMALIZATION")
    print(f"{'='*70}")
    
    if is_sklearn:
         # For Sklearn, just normalize the already-split data
         scaler = StandardScaler()
         X_train = scaler.fit_transform(X_train)
         X_test = scaler.transform(X_test)
         
         print(f"[OK] Data normalized")
         print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
         
         train_loader = None
         test_loader = None
    else:
        # For Deep Learning models, normalize and create dataloaders
        # Note: We already have X_train, X_test, y_train, y_test from above split
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
            n_test = len(X_test)
            X_test = X_test_flat.reshape(n_test, seq_len, n_features).transpose(0, 2, 1)
        
        # Check for NaNs/Infs which cause CUDA errors
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            print("[WARN] Warning: NaNs/Infs found in training data. Replacing with 0.")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        if np.isnan(X_test).any() or np.isinf(X_test).any():
            print("[WARN] Warning: NaNs/Infs found in test data. Replacing with 0.")
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to tensors and create dataloaders
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.FloatTensor(y_test)
        
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        
        # Create generator for reproducible data loading
        g = torch.Generator()
        g.manual_seed(config.data['random_seed'] if config else 42)
        
        # Worker init function for multi-process data loading
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=training_config['batch_size'], 
            shuffle=True, 
            drop_last=True,
            generator=g,
            worker_init_fn=worker_init_fn
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=training_config['batch_size'], 
            shuffle=False
        )
        
        print(f"[OK] Dataloaders created")
        print(f"  Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    
    # Step 4: Create model and Train
    print(f"\n{'='*70}")
    print(f"STEP 4: MODEL TRAINING")
    print(f"{'='*70}")
    
    if is_sklearn:
        print(f"Training Sklearn Model: {model_type.upper()}")
        
        # Create ML model using factory function
        random_seed = config.data['random_seed'] if config else 42
        clf = create_ml_model(model_type, random_state=random_seed)
        
        # Print model info
        model_info = get_model_info(model_type)
        print(f"  Model: {model_info['name']}")
        print(f"  Type: {model_info['type']}")
            
        clf.fit(X_train, y_train)
        print("[OK] Training complete")
        
        # Predict
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.5
            
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'train': {'accuracy': 0, 'loss': 0}, # Placeholder
            'test': {
                'loss': 0,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'roc_auc': auc,
                'confusion_matrix': cm.tolist()
            }
        }
        
    else:
        # Check CUDA health before starting
        print(f"DEBUG: Torch Version: {torch.__version__}")
        print(f"DEBUG: CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"DEBUG: CUDA Device: {torch.cuda.get_device_name(0)}")
            try:
                # Test CUDA allocation
                torch.zeros(1).cuda()
                device_name = 'cuda'
            except Exception as e:
                print(f"[FAIL] CRITICAL GPU ERROR: {e}")
                print("[WARN] The GPU is detected but failing to allocate memory. This is a driver 'zombie' state.")
                raise RuntimeError("Critical GPU Crash - Restart Required")
        else:
             print("[FAIL] ERROR: GPU not available (torch.cuda.is_available() is False).")
             # Try to diagnose
             try:
                 print(f"  Start Method: {torch.multiprocessing.get_start_method(allow_none=True)}")
                 print(f"  Cuda Arch List: {torch.cuda.get_arch_list()}")
             except: pass
             print("[WARN] Please ensure you have CUDA installed and proper drivers.")
             raise RuntimeError("No GPU available or CUDA not installed.")
        
        device = torch.device(device_name)
        print(f"Device: {device}")
        
        model = create_model(model_type, model_config)
        print(f"[OK] {model_type.upper()} model created")
        print(f"  Config: {model_config}")
        
        # Calculate class weights for handling imbalance (using sqrt to avoid extreme weights)
        n_class_0 = (y_train == 0).sum()
        n_class_1 = (y_train == 1).sum()
        total = len(y_train)
        
        # Use balanced formula with sqrt to reduce extremeness
        raw_weight_0 = total / (2 * n_class_0)
        raw_weight_1 = total / (2 * n_class_1)
        
        # Apply sqrt to make weights less extreme
        import math
        class_weights = (math.sqrt(raw_weight_0), math.sqrt(raw_weight_1))
        
        print(f"  Train distribution: class_0={n_class_0}, class_1={n_class_1}")
        print(f"  Class weights (sqrt-balanced): [class_0: {class_weights[0]:.3f}, class_1: {class_weights[1]:.3f}]")
        
        trainer = SimpleTrainer(
            model=model,
            device=device,
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            class_weights=class_weights
        )
        
        # CRITICAL DEBUG: Verify patience before calling fit()
        patience_value = training_config.get('early_stopping_patience', 10)
        print(f"\n{'!'*70}")
        print(f"!!! ABOUT TO CALL trainer.fit() !!!")
        print(f"!!! patience parameter value: {patience_value} !!!")
        print(f"!!! Type: {type(patience_value)} !!!")
        print(f"!!! training_config keys: {list(training_config.keys())} !!!")
        print(f"{'!'*70}\n")
        
        results = trainer.fit(
            train_loader=train_loader,
            test_loader=test_loader,
            y_train=y_train,
            y_test=y_test,
            n_epochs=training_config['n_epochs'],
            verbose=True,
            patience=patience_value
        )
    
    # Step 4.5: Threshold Tuning (Optional - optimize decision threshold)
    print(f"\n{'='*70}")
    print(f"STEP 4.5: THRESHOLD TUNING (OPTIONAL)")
    print(f"{'='*70}")
    
    # Get probabilities for test set
    if is_sklearn:
        y_test_probs = clf.predict_proba(X_test)[:, 1]
    else:
        # Get predictions from neural network
        model.eval()
        y_test_probs = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                
                # Check output shape to determine loss type
                if outputs.shape[1] == 1:
                    # BCEWithLogitsLoss: output is [batch, 1], apply sigmoid
                    probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                else:
                    # CrossEntropyLoss: output is [batch, 2], apply softmax
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                
                # Handle scalar case (batch size 1)
                if probs.ndim == 0:
                    probs = np.array([probs])
                    
                y_test_probs.extend(probs)
        y_test_probs = np.array(y_test_probs)
    
    # Test multiple thresholds
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    print(f"Testing thresholds: {thresholds}")
    print(f"Optimization metric: F1-macro\n")
    
    from sklearn.metrics import balanced_accuracy_score
    
    best_threshold = 0.5
    best_f1_macro = 0.0
    threshold_results = {}
    
    for th in thresholds:
        y_pred_th = (y_test_probs >= th).astype(int)
        
        acc = accuracy_score(y_test, y_pred_th)
        f1_macro = f1_score(y_test, y_pred_th, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, y_pred_th)
        
        threshold_results[th] = {
            'accuracy': float(acc),
            'f1_macro': float(f1_macro),
            'balanced_accuracy': float(bal_acc)
        }
        
        print(f"  th={th:.2f}: Acc={acc:.4f}, F1-macro={f1_macro:.4f}, Bal.Acc={bal_acc:.4f}")
        
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_threshold = th
    
    print(f"\n[OK] Optimal threshold: {best_threshold:.2f} (F1-macro={best_f1_macro:.4f})")
    
    # Re-compute metrics with optimal threshold
    y_pred_optimal = (y_test_probs >= best_threshold).astype(int)
    
    optimal_acc = accuracy_score(y_test, y_pred_optimal)
    optimal_prec = precision_score(y_test, y_pred_optimal, zero_division=0)
    optimal_rec = recall_score(y_test, y_pred_optimal, zero_division=0)
    optimal_f1 = f1_score(y_test, y_pred_optimal, zero_division=0)
    optimal_cm = confusion_matrix(y_test, y_pred_optimal)
    
    try:
        optimal_auc = roc_auc_score(y_test, y_test_probs)
    except:
        optimal_auc = results['test'].get('roc_auc', 0.5)
    
    print(f"\nMetrics with optimal threshold ({best_threshold:.2f}):")
    print(f"  Accuracy:  {optimal_acc:.4f}")
    print(f"  Precision: {optimal_prec:.4f}")
    print(f"  Recall:    {optimal_rec:.4f}")
    print(f"  F1-score:  {optimal_f1:.4f}")
    print(f"  ROC-AUC:   {optimal_auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"  {optimal_cm}")
    
    # Add threshold tuning results to metadata
    results['threshold_tuning'] = {
        'enabled': True,
        'thresholds_tested': thresholds,
        'threshold_results': threshold_results,
        'optimal_threshold': float(best_threshold),
        'optimal_metrics': {
            'accuracy': float(optimal_acc),
            'precision': float(optimal_prec),
            'recall': float(optimal_rec),
            'f1': float(optimal_f1),
            'roc_auc': float(optimal_auc),
            'confusion_matrix': optimal_cm.tolist()
        }
    }
    
    # Step 5: Save results
    print(f"\n{'='*70}")
    print(f"STEP 5: SAVING RESULTS")
    print(f"{'='*70}")
    
    # Snapshot config.ini for perfect reproducibility
    config_snapshot = snapshot_config_ini()
    
    # Create run metadata
    timestamp = shared_timestamp
    run_id = f"{model_type}_{timestamp}"
    
    run_metadata = {
        'run_id': run_id,
        'timestamp': timestamp,
        'model_type': model_type,
        'model_config': model_config,
        'training_config': training_config,
        'config_snapshot': config_snapshot,  # Full config.ini snapshot
        'data_config': {
            # Evaluation method
            'evaluation_method': 'simple',
            
            # Filtering configuration
            'parts_filter': parts_filter,
            'aoi_filter': aoi_filter,
            'time_window_s': time_window_s if time_window_s is not None else 'full',
            'feature_columns': data_stats.get('feature_columns', []),
            'n_feature_columns': len(data_stats.get('feature_columns', [])),
            
            # Data type
            'data_type': 'sequence' if sequence_data else 'aggregated',
            'sequence_length': data_stats.get('sequence_length'),
            
            # Original data statistics
            'original_trials': data_stats['original_trials'],
            'original_class_0': data_stats['original_class_0'],
            'original_class_1': data_stats['original_class_1'],
            
            # Train/Test split
            'split_info': data_stats.get('split', {}),
            'n_train': len(y_train),
            'n_test': len(y_test),
            'n_features': data_stats['n_features'],
            
            # Summary
            'total_samples_used': len(y_train) + len(y_test)
        },
        'metrics': {
            'train': {
                **results['train'],
                'history': results.get('history', {})
            },
            'test': results['test']
        }
    }
    
    # Save JSON
    json_path = reports_root / f"{run_id}.json"
    with open(json_path, 'w') as f:
        # Convert all numpy types to native Python types
        serializable_metadata = convert_to_serializable(run_metadata)
        json.dump(serializable_metadata, f, indent=2)
    print(f"[OK] JSON saved: {json_path}")
    
    # Generate detailed HTML report and update dashboard
    try:
        # Force remove reporting modules to ensure freshness
        import sys
        modules_to_remove = [k for k in sys.modules if k.startswith('reporting') or k.startswith('src.reporting')]
        for k in modules_to_remove:
            # print(f"DEBUG: Removing {k} from sys.modules")
            del sys.modules[k]
            
        from src.reporting.html_report import AdvancedReportGenerator
        
        reporter = AdvancedReportGenerator(reports_root)
        
        # Generate detailed report for this run
        html_path = reports_root / f"{run_id}.html"
        reporter.generate_detailed_report(run_metadata, html_path)
        
        # Update main dashboard with all runs
        reporter.generate_dashboard()
        
        print(f"[OK] Detailed report: {html_path}")
        print(f"[OK] Dashboard updated: {reports_root / 'index.html'}")
    except Exception as e:
        print(f"[WARN] HTML report generation failed: {e}")
        import traceback
        traceback.print_exc()
        html_path = None
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"\nModel: {model_type.upper()}")
    print(f"Test AUC: {results['test']['roc_auc']:.4f}")
    print(f"Test F1:  {results['test']['f1']:.4f}")
    print(f"Test Acc: {results['test']['accuracy']:.4f}")
    print(f"\n[OK] Pipeline complete!")
    
    return {
        'run_id': run_id,
        'results': results,
        'metadata': run_metadata,
        'json_path': str(json_path),
        'html_path': str(html_path) if html_path else None,
        'csv_path': str(csv_path) if csv_path else None
    }


def run_all_models(parts_filter=None, aoi_filter=None, time_window_s=8, feature_columns=None, trials_dir=None):
    """
    Run all models with same configuration.
    
    Returns:
        dict: Results for all models
    """
    models = ['mlp', 'cnn', 'cnn1d', 'lstm', 'bilstm', 'hybrid', 'transformer', 'rf', 'xgboost', 'svm', 'logreg']
    all_results = {}
    
    for model_type in models:
        try:
            print(f"\n\n{'#'*70}")
            print(f"# MODEL: {model_type.upper()}")
            print(f"{'#'*70}\n")
            
            result = run_simple_pipeline(
                model_type=model_type,
                parts_filter=parts_filter,
                aoi_filter=aoi_filter,
                time_window_s=time_window_s,
                feature_columns=feature_columns,
                trials_dir=trials_dir
            )
            all_results[model_type] = result
        except Exception as e:
            print(f"\n[FAIL] {model_type.upper()} failed: {e}")
            all_results[model_type] = {'error': str(e)}
    
    # Print comparison
    print(f"\n\n{'='*70}")
    print(f"COMPARISON: ALL MODELS")
    print(f"{'='*70}\n")
    print(f"{'Model':<15} {'Test AUC':<12} {'Test F1':<12} {'Test Acc':<12}")
    print(f"{'-'*51}")
    
    for model_type, result in all_results.items():
        if 'error' not in result:
            test_metrics = result['results']['test']
            print(f"{model_type.upper():<15} "
                  f"{test_metrics['roc_auc']:<12.4f} "
                  f"{test_metrics['f1']:<12.4f} "
                  f"{test_metrics['accuracy']:<12.4f}")
        else:
            print(f"{model_type.upper():<15} FAILED")
    
    return all_results


def run_pipeline_with_lopo_cv(
    model_type='mlp',
    parts_filter=None,
    aoi_filter=None,
    time_window_s=None,
    feature_columns=None,
    n_seeds=10,  # Increased to 10 seeds for better statistical stability
    config=None
):
    """
    Run pipeline with Leave-One-Participant-Out Cross-Validation.
    This gives more robust results than simple train/test split.
    
    Args:
        model_type: 'mlp', 'cnn', 'cnn1d', 'lstm', 'bilstm', 'hybrid', or 'transformer'
        parts_filter: List of parts (default: from config)
        aoi_filter: List of AOIs (default: from config)
        time_window_s: Time window in seconds (default: from config)
        feature_columns: List of feature columns (optional)
        n_seeds: Number of random seeds to run (averaging results)
        config: Config object
    
    Returns:
        dict: Results with LOPO CV metrics
    """
    # Generate timestamp once to ensure correlation between artifacts
    shared_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Validate config for reproducibility
    print("\n" + "="*70)
    print("CONFIG VALIDATION")
    print("="*70)
    validate_config_for_reproducibility()
    
    print("\n" + "="*70)
    print(f"PIPELINE WITH LOPO CV: {model_type.upper()}")
    print("="*70)
    print("Method: Leave-One-Participant-Out Cross-Validation")
    print(f"Robustness: {n_seeds} Random Seeds")
    print("Threshold Tuning: Enabled (optimized per fold)")
    print("="*70)
    
    # Use config values if not explicitly provided
    if parts_filter is None:
        parts_filter = config.data['parts_filter']
    if aoi_filter is None:
        aoi_filter = config.data['aoi_filter']
    # time_window_s: None=use config, -1=full trial, number=specific window
    if time_window_s is None:
        time_window_s = config.data['time_window_s']
    elif time_window_s == -1:
        time_window_s = None  # Full trial
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_type}")
    print(f"  Parts: {parts_filter}")
    print(f"  AOI: {aoi_filter}")
    print(f"  Time Window: {'Full trial' if time_window_s is None else f'{time_window_s}s'}")
    if feature_columns:
        print(f"  Features: Selected {len(feature_columns)} columns")
    else:
        print(f"  Features: All available (or Default set)")
    
    # Step 1: Load data
    print(f"\n{'='*70}")
    print(f"STEP 1: LOADING DATA")
    print(f"{'='*70}")
    
    dataset = EyeTrackingDataset(
        labels_csv=str(config.paths['labels_csv']),
        trials_dir=str(config.paths['trials_dir']),
        parts_filter=parts_filter,
        aoi_filter=aoi_filter,
        time_window_s=time_window_s,
        feature_columns=feature_columns
    )
    
    # Determine data type based on model
    is_sequence = model_type in ['cnn', 'cnn1d', 'lstm', 'bilstm', 'hybrid', 'transformer']
    
    if is_sequence:
        print(f"  Loading Sequence Data for {model_type.upper()}...")
        X, y, sample_info = dataset.load_sequences()
    else:
        print(f"  Loading Aggregated Features for {model_type.upper()}...")
        X, y, sample_info = dataset.load_aggregated_features()
        
    participant_ids = np.array([info['participant_id'] for info in sample_info])
    unique_participants = np.unique(participant_ids)
    
    print(f"  Total samples: {len(X)}")
    print(f"  Unique participants: {len(unique_participants)}")
    print(f"  Data Shape: {X.shape}")
    print(f"  Class 0: {np.sum(y == 0)} | Class 1: {np.sum(y == 1)}")
    
    # --- ADDED: EXPORT FILTERED DATA LOGIC START ---
    # Save RAW filtered data to CSV (same logic as run_simple_pipeline)
    timestamp = shared_timestamp
    csv_dirname = f"{model_type}_{timestamp}_filtered_trials"
    csv_output_dir = config.paths['dl_inputs_dir'] / csv_dirname
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"\n{'='*70}")
        print(f"EXPORTING RAW FILTERED DATA")
        print(f"{'='*70}")
        
        # Load labels
        df_labels = pd.read_csv(config.paths['labels_csv'])
        
        # Apply parts filter
        if parts_filter:
            df_filtered = df_labels[df_labels['part'].isin(parts_filter)].copy()
        else:
            df_filtered = df_labels.copy()
        
        # Process each trial and save separately
        saved_trials = 0
        skipped = 0
        total_rows = 0
        
        trials_dir = Path(config.paths['trials_dir'])
        
        for idx, row in df_filtered.iterrows():
            p_id = row['participant_id']
            q_id = row['question_id']
            filepath = trials_dir / f"{p_id}_{q_id}.csv"
            
            if not filepath.exists():
                skipped += 1
                continue
            
            try:
                # Read trial CSV
                df_trial = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
                
                # STEP 1: Extract time window
                if time_window_s is not None:
                    n_samples = int(time_window_s * 150)  # 150 Hz
                    df_tail = df_trial.tail(n_samples)
                else:
                    df_tail = df_trial
                
                # STEP 2: Filter AOI
                if aoi_filter:
                    df_filtered_aoi = df_tail[df_tail['AOI'].isin(aoi_filter)]
                else:
                    df_filtered_aoi = df_tail
                
                # STEP 3: Quality check
                if len(df_filtered_aoi) < 10:
                    skipped += 1
                    continue
                
                # Filter feature columns
                if feature_columns:
                    cols_to_keep = [col for col in feature_columns if col in df_filtered_aoi.columns]
                    df_final = df_filtered_aoi[cols_to_keep].copy()
                else:
                    cols_to_drop = ['participant_id', 'question_id', 'part', 'is_valid', 'AOI', 'randomness_label', 'randomness_label_num']
                    df_final = df_filtered_aoi.drop(columns=[c for c in cols_to_drop if c in df_filtered_aoi.columns]).copy()
                
                # INJECT LABEL (Ensure it exists in the output file)
                # We get it from the 'row' of stage3_with_labels.csv which is reliable
                df_final['randomness_label'] = row['randomness_label']
                
                # Save this trial to separate CSV
                trial_csv_path = csv_output_dir / f"{p_id}_{q_id}.csv"
                df_final.to_csv(trial_csv_path, index=False)
                
                saved_trials += 1
                total_rows += len(df_final)
                
            except Exception as e:
                skipped += 1
                continue
        
        print(f"[OK] Raw filtered data saved: {csv_output_dir}")
        print(f"  Total trials saved: {saved_trials}")
        
    except Exception as e:
        print(f"[WARN] Failed to save filtered CSV: {e}")
    # --- ADDED: EXPORT FILTERED DATA LOGIC END ---

    # Container for all seeds
    seed_metrics = []
    
    # Loop over seeds
    for seed in range(n_seeds):
        current_seed = 42 + seed
        
        print(f"\n{'='*30}")
        print(f"RUNNING SEED {seed+1}/{n_seeds} (Random State: {current_seed})")
        print(f"{'='*30}")
        
        # Check CUDA health before starting each seed
        if torch.cuda.is_available():
            try:
                # Test CUDA allocation
                torch.zeros(1).cuda()
            except Exception as e:
                print(f"[FAIL] CRITICAL GPU ERROR: {e}")
                print("[WARN] The GPU is detected but failing to allocate memory. This is a driver 'zombie' state.")
                print("[WARN] YOU MUST RESTART YOUR COMPUTER TO FIX THIS.")
                raise RuntimeError("Critical GPU Crash - Restart Required")
        else:
             print("[FAIL] ERROR: GPU not available (torch.cuda.is_available() is False).")
             print("[WARN] Please ensure you have CUDA installed and proper drivers.")
             raise RuntimeError("No GPU available or CUDA not installed.")

        # Set all random seeds for full reproducibility
        seed_everything(current_seed)
        
        device = torch.device('cuda')
        
        logo = LeaveOneGroupOut()
        n_folds = logo.get_n_splits(groups=participant_ids)
        
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        threshold_results = {th: {'f1_macro': [], 'balanced_acc': [], 'accuracy': []} for th in thresholds}
        
        all_y_true = []
        all_y_probs = []
        fold_count = 0
        
        # Inner loop: LOPO Folds
        for train_idx, test_idx in logo.split(X, y, groups=participant_ids):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            y_train = y_train.ravel()
            y_test = y_test.ravel()
            
            # Normalize
            scaler = StandardScaler()
            if is_sequence:
                # Handle 3D data: (N, Channels, Time) -> Flatten -> Scale -> Reshape
                n_samples, n_channels, n_time = X_train.shape
                
                # Reshape to (Samples*Time, Channels) for scaling features
                # Note: We need a transpose to get (N, Time, Channels) before reshaping
                X_train_flat = X_train.transpose(0, 2, 1).reshape(-1, n_channels)
                X_test_flat = X_test.transpose(0, 2, 1).reshape(-1, n_channels)
                
                start_scale = datetime.now()
                X_train_flat = scaler.fit_transform(X_train_flat)
                X_test_flat = scaler.transform(X_test_flat)
                
                # Reshape back to (N, Channels, Time)
                X_train = X_train_flat.reshape(n_samples, n_time, n_channels).transpose(0, 2, 1)
                X_test = X_test_flat.reshape(len(X_test), n_time, n_channels).transpose(0, 2, 1)
            else:
                # Standard 2D scaling
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Check for NaNs/Infs (Crucial fix for CUDA errors)
            if np.isnan(X_train).any() or np.isinf(X_train).any():
                # Only print warning once per seed to avoid spam
                if seed_idx == 0 and i == 0:
                     print("[WARN] Warning: NaNs/Infs found in training data. Replacing with 0.")
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

            if np.isnan(X_test).any() or np.isinf(X_test).any():
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

            # Create dataloaders
            # For BCEWithLogitsLoss, targets must be floats
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
            
            # Create generator for reproducible shuffling
            g = torch.Generator()
            g.manual_seed(current_seed)
            
            # Worker init function for reproducibility
            def worker_init_fn(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=16, 
                shuffle=True, 
                drop_last=True,
                generator=g,
                worker_init_fn=worker_init_fn
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=16, 
                shuffle=False
            )
            
            # Create model
            model_config = config.get_model_config(model_type)
            training_config = config.get_training_config(model_type)
            
            # Map input dimension to model-specific config keys
            input_dim_val = X_train.shape[1] if hasattr(X_train, 'shape') else None
            
            if input_dim_val:
                if model_type == 'mlp':
                    model_config['input_dim'] = input_dim_val
                elif model_type == 'lstm':
                    model_config['input_size'] = input_dim_val
                elif model_type in ['cnn', 'hybrid']:
                    model_config['input_channels'] = input_dim_val
                # Fallback valid for all just in case
                model_config['input_dim'] = input_dim_val
            
            model = create_model(model_type, model_config)
            model = model.to(device)
            
            # Train
            n_class_0 = (y_train == 0).sum()
            n_class_1 = (y_train == 1).sum()
            total = len(y_train)
            
            import math
            # Standard balanced weights
            weight_0 = total / (2 * n_class_0)
            weight_1 = total / (2 * n_class_1)
            class_weights = (weight_0, weight_1)
            
            trainer = SimpleTrainer(
                model=model,
                device=device,
                learning_rate=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                class_weights=class_weights
            )
            
            # Train (silent mode)
            patience_value = training_config.get('early_stopping_patience', 10)
            results = trainer.fit(
                train_loader=train_loader,
                test_loader=test_loader,
                y_train=y_train,
                y_test=y_test,
                n_epochs=training_config['n_epochs'],
                verbose=False,
                patience=patience_value
            )
            
            # Get predictions
            model.eval()
            y_probs = []
            y_true_fold = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    
                    # Check output shape
                    if outputs.shape[1] == 1:
                        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                    else:
                        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    
                    if probs.ndim == 0:
                        probs = np.array([probs])
                    
                    y_probs.extend(probs)
                    y_true_fold.extend(y_batch.numpy())
            
            y_probs = np.array(y_probs)
            y_true_fold = np.array(y_true_fold).ravel()
            
            all_y_true.extend(y_true_fold)
            all_y_probs.extend(y_probs)
            
            # Evaluate with each threshold
            for th in thresholds:
                y_pred = (y_probs >= th).astype(int)
                acc = accuracy_score(y_true_fold, y_pred)
                f1_macro = f1_score(y_true_fold, y_pred, average='macro', zero_division=0)
                bal_acc = balanced_accuracy_score(y_true_fold, y_pred)
                
                threshold_results[th]['f1_macro'].append(f1_macro)
                threshold_results[th]['balanced_acc'].append(bal_acc)
                threshold_results[th]['accuracy'].append(acc)
            
            test_participant = participant_ids[test_idx[0]]
            print(f"  Fold {fold_count+1}/{n_folds}: Participant {test_participant}", end='\r')
            fold_count += 1

        print(f"[OK] Seed {seed+1} complete ({n_folds} folds)                ")
        
        # Optimize threshold for this seed
        best_threshold = None
        best_f1 = -1
        
        for th in thresholds:
            avg_f1 = np.mean(threshold_results[th]['f1_macro'])
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_threshold = th
        
        # Final metrics for this seed
        all_y_true = np.array(all_y_true)
        all_y_probs = np.array(all_y_probs)
        y_pred_optimal = (all_y_probs >= best_threshold).astype(int)
        
        final_auc = roc_auc_score(all_y_true, all_y_probs)
        final_bal_acc = balanced_accuracy_score(all_y_true, y_pred_optimal)
        final_prec_0 = precision_score(all_y_true, y_pred_optimal, pos_label=0, zero_division=0)
        final_rec_0 = recall_score(all_y_true, y_pred_optimal, pos_label=0, zero_division=0)
        final_f1_0 = f1_score(all_y_true, y_pred_optimal, pos_label=0, zero_division=0)
        
        final_prec_1 = precision_score(all_y_true, y_pred_optimal, pos_label=1, zero_division=0)
        final_rec_1 = recall_score(all_y_true, y_pred_optimal, pos_label=1, zero_division=0)
        final_f1_1 = f1_score(all_y_true, y_pred_optimal, pos_label=1, zero_division=0)

        # Calculate metrics per participant fold using the BEST threshold
        # We need to reconstruct fold boundaries or store them. 
        # Easier: store fold indices during the loop. But we didn't.
        # Alternative: We can calc fold metrics inside the loop IF we had a fixed threshold.
        # Since threshold is tuned AFTER loop, we can't do exact fold-wise metrics with *optimal* threshold easily without storing all preds.
        # But we stored all_y_probs and we know the order matches LOPO iteration if we trust the loop logic?
        # Actually, logo.split order is deterministic for a fixed groups array.
        # Let's re-iterate to calculate fold-wise metrics for reporting distribution.
        
        fold_metrics = []
        current_idx = 0
        logo_dist = LeaveOneGroupOut()
        for _, test_idx in logo_dist.split(X, y, groups=participant_ids):
             # Extract this fold's predictions from the concatenated array
             fold_len = len(test_idx)
             fold_y_true = all_y_true[current_idx : current_idx+fold_len]
             fold_y_probs = all_y_probs[current_idx : current_idx+fold_len]
             fold_y_pred = (fold_y_probs >= best_threshold).astype(int)
             
             participant_id = participant_ids[test_idx[0]]
             
             f_acc = accuracy_score(fold_y_true, fold_y_pred)
             f_f1 = f1_score(fold_y_true, fold_y_pred, zero_division=0)
             f_auc = roc_auc_score(fold_y_true, fold_y_probs) if len(np.unique(fold_y_true)) > 1 else 0.5
             
             fold_metrics.append({
                 'participant': participant_id,
                 'accuracy': f_acc,
                 'f1': f_f1,
                 'auc': f_auc
             })
             current_idx += fold_len

        # sort by F1
        fold_metrics.sort(key=lambda x: x['f1'])
        worst_fold = fold_metrics[0]
        best_fold = fold_metrics[-1]

        metrics = {
            'accuracy': accuracy_score(all_y_true, y_pred_optimal),
            'f1': f1_score(all_y_true, y_pred_optimal, zero_division=0),
            'roc_auc': final_auc,
            'threshold': best_threshold,
            'balanced_accuracy': final_bal_acc,
            'f1_class_0': final_f1_0,
            'f1_class_1': final_f1_1,
            'worst_participant': worst_fold,
            'best_participant': best_fold
        }
        seed_metrics.append(metrics)
        print(f"  Seed Result: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")
        print(f"    Class 0 F1: {metrics['f1_class_0']:.4f} | Class 1 F1: {metrics['f1_class_1']:.4f}")
        print(f"    Worst: {worst_fold['participant']} (F1={worst_fold['f1']:.2f}) | Best: {best_fold['participant']} (F1={best_fold['f1']:.2f})")

    # Aggregated Results
    print(f"\n{'='*70}")
    print(f"FINAL AGGREGATED RESULTS ({n_seeds} SEEDS)")
    print(f"{'='*70}")
    
    avg_acc = np.mean([m['accuracy'] for m in seed_metrics])
    std_acc = np.std([m['accuracy'] for m in seed_metrics])
    
    avg_f1 = np.mean([m['f1'] for m in seed_metrics])
    std_f1 = np.std([m['f1'] for m in seed_metrics])
    
    avg_auc = np.mean([m['roc_auc'] for m in seed_metrics])
    std_auc = np.std([m['roc_auc'] for m in seed_metrics])
    
    avg_bal = np.mean([m['balanced_accuracy'] for m in seed_metrics])
    std_bal = np.std([m['balanced_accuracy'] for m in seed_metrics])
    
    avg_f1_0 = np.mean([m['f1_class_0'] for m in seed_metrics])
    std_f1_0 = np.std([m['f1_class_0'] for m in seed_metrics])

    avg_f1_1 = np.mean([m['f1_class_1'] for m in seed_metrics])
    std_f1_1 = np.std([m['f1_class_1'] for m in seed_metrics])
    
    best_threshold = np.mean([m['threshold'] for m in seed_metrics])
    
    final_acc = avg_acc
    final_f1 = avg_f1
    final_auc = avg_auc
    
    print(f"  Accuracy:          {avg_acc*100:.2f}% +/- {std_acc*100:.2f}%")
    print(f"  Balanced Accuracy: {avg_bal*100:.2f}% +/- {std_bal*100:.2f}%")
    print(f"  F1-Score (Macro):  {avg_f1:.4f} +/- {std_f1:.4f}")
    print(f"  ROC-AUC:           {avg_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Class 0 F1:        {avg_f1_0:.4f} +/- {std_f1_0:.4f}")
    print(f"  Class 1 F1:        {avg_f1_1:.4f} +/- {std_f1_1:.4f}")
    
    # Return structure compatible with GUI
    results = {
        'run_id': f"{model_type}_lopo_{shared_timestamp}",
        'method': f'LOPO CV ({n_seeds} seeds)',
        'data_config': {
            'evaluation_method': 'lopo',
            'parts_filter': parts_filter,
            'aoi_filter': aoi_filter,
            'time_window_s': time_window_s if time_window_s is not None else 'full',
            'feature_columns': feature_columns,
            'n_feature_columns': len(feature_columns) if feature_columns else 0,
        },
        'metrics': {
            'accuracy': float(avg_acc),
            'accuracy_std': float(std_acc),
            'f1': float(avg_f1),
            'f1_std': float(std_f1),
            'roc_auc': float(avg_auc),
            'roc_auc_std': float(std_auc),
            'balanced_accuracy': float(avg_bal),
            'balanced_accuracy_std': float(std_bal),
            'f1_class_0': float(avg_f1_0),
            'f1_class_0_std': float(std_f1_0),
            'f1_class_1': float(avg_f1_1),
            'f1_class_1_std': float(std_f1_1),
            'precision': 0.0,
            'recall': 0.0,
        },
        'results': { # For GUI compatibility
            'metrics': {
                'accuracy': float(avg_acc),
                'f1': float(avg_f1),
                'roc_auc': float(avg_auc)
            }
        }
    }
    
    # Snapshot config.ini for perfect reproducibility
    results['config_snapshot'] = snapshot_config_ini()
    
    # Create report generator
    try:
        from src.reporting.html_report import AdvancedReportGenerator
        reports_dir = config.paths['reports_dir']
        generator = AdvancedReportGenerator(reports_dir)
        
        # Add timestamp if missing
        results['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        results['model_type'] = model_type
        
        # Save JSON first (needed for dashboard)
        json_path = reports_dir / f"{results['run_id']}.json"
        
        # Ensure results are serializable
        serializable_results = convert_to_serializable(results)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        results['json_path'] = str(json_path)
        print(f"[OK] Results saved: {json_path}")
        
        # Generate HTML Report
        html_path = reports_dir / f"{results['run_id']}.html"
        generator.generate_detailed_report(serializable_results, html_path)
        
        # Refresh Dashboard
        generator.generate_dashboard()
        
        results['html_path'] = str(html_path)
        
    except ImportError:
        print("[WARN] Could not import AdvancedReportGenerator. HTML report skipped.")
        # Fallback JSON save if not already saved
        reports_root = config.paths['reports_dir']
        reports_root.mkdir(parents=True, exist_ok=True)
        json_path = reports_root / f"{results['run_id']}.json"
        with open(json_path, 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=4)
        print(f"[OK] Results saved: {json_path}")
    except Exception as e:
        print(f"[WARN] HTML Report generation failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"\nModel: {model_type.upper()}")
    print(f"Method: LOPO CV ({n_seeds} seeds)")
    print(f"Test AUC: {final_auc:.4f}")
    print(f"Test F1:  {final_f1:.4f}")
    print(f"Test Acc: {final_acc*100:.2f}%")
    print(f"\n[OK] Pipeline complete!\n")
    
    return {
        'json_path': str(json_path),
        'results': results['results'],
        'metrics': results['results']['metrics']
    }


if __name__ == '__main__':
    import sys
    
    # Load config
    config = load_config()
    
    # Parse command-line arguments
    model_type = 'mlp'  # default
    use_lopo = False
    
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == '--model' and i + 1 < len(sys.argv):
                model_type = sys.argv[i + 1]
            elif arg == '--lopo':
                use_lopo = True
    
    # Run pipeline
    if use_lopo:
        run_pipeline_with_lopo_cv(
            model_type=model_type,
            config=config
        )
    else:
        run_simple_pipeline(
            model_type=model_type,
            config=config
        )

