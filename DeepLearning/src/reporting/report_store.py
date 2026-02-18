"""Report store for managing run metadata."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class ReportStore:
    """Manage run metadata storage and retrieval."""
    
    def __init__(self, reports_root: Path):
        """
        Initialize report store.
        
        Args:
            reports_root: Root directory for reports
        """
        self.reports_root = Path(reports_root)
        self.metadata_file = self.reports_root / 'runs_metadata.json'
        
        # Ensure directory exists
        self.reports_root.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_run(
        self,
        experiment_id: str,
        run_id: str,
        config: Dict,
        dataset_stats: Dict,
        training_results: Dict,
        ui_selections: Dict,
        artifacts_dir: str = None
    ):
        """
        Add a run to the store.
        
        Args:
            experiment_id: Experiment ID
            run_id: Run ID
            config: Configuration dictionary
            dataset_stats: Dataset statistics
            training_results: Training results
            ui_selections: UI selections (experiment_id, aoi_mode, window_s, run_note)
            artifacts_dir: Optional explicit artifacts directory path
        """
        if experiment_id not in self.metadata:
            self.metadata[experiment_id] = {}
        
        # Use provided artifacts_dir or compute default
        if artifacts_dir is None:
            artifacts_dir = str(self.reports_root.parent / 'dl_inputs' / 'experiments' / experiment_id / run_id)
        
        self.metadata[experiment_id][run_id] = {
            'timestamp': datetime.now().isoformat(),
            'ui_selections': ui_selections,
            'config': config,
            'dataset_stats': dataset_stats,
            'training_results': {
                'best_epoch': training_results['best_epoch'],
                'best_val_metric': training_results['best_val_metric'],
                'test_metrics': training_results['test_metrics'],
                'early_stopped': training_results['early_stopped'],
                'total_epochs': training_results['total_epochs'],
            },
            'artifacts_dir': artifacts_dir,
        }
        
        self._save_metadata()
    
    def get_experiment_runs(self, experiment_id: str) -> List[Dict]:
        """
        Get all runs for an experiment, sorted by run_id numerically.
        
        Args:
            experiment_id: Experiment ID
        
        Returns:
            List of run dictionaries with run_id included, sorted chronologically
        """
        if experiment_id not in self.metadata:
            return []
        
        runs = []
        for run_id, run_data in self.metadata[experiment_id].items():
            run_data_copy = run_data.copy()
            run_data_copy['run_id'] = run_id
            runs.append(run_data_copy)
        
        # Sort runs numerically by run_id (handle both run_XXXX and run_XXXX_modeltype)
        def extract_run_number(run_id):
            parts = run_id.split('_')
            if len(parts) >= 2:
                try:
                    return int(parts[1])  # run_0011_baseline -> 11
                except ValueError:
                    return int(parts[-1])  # fallback for old format
            return 0
        
        runs.sort(key=lambda r: extract_run_number(r['run_id']))
        
        return runs
    
    def get_all_experiments(self) -> List[str]:
        """Get list of all experiment IDs."""
        return list(self.metadata.keys())
    
    def get_best_run_per_experiment(self, primary_metric: str) -> Dict[str, Dict]:
        """
        Get best run for each experiment based on primary metric.
        
        Args:
            primary_metric: Metric to use for comparison
        
        Returns:
            Dictionary mapping experiment_id to best run data
        """
        best_runs = {}
        
        for exp_id in self.get_all_experiments():
            runs = self.get_experiment_runs(exp_id)
            
            if not runs:
                continue
            
            # Find best run
            best_run = max(
                runs,
                key=lambda r: r['training_results']['test_metrics'].get(primary_metric, -np.inf)
            )
            best_runs[exp_id] = best_run
        
        return best_runs
    
    def get_run_comparison_df(self, experiment_id: str, primary_metric: str) -> pd.DataFrame:
        """
        Get dataframe comparing runs for an experiment.
        
        Args:
            experiment_id: Experiment ID
            primary_metric: Primary metric for sorting
        
        Returns:
            DataFrame with run comparison
        """
        runs = self.get_experiment_runs(experiment_id)
        
        if not runs:
            return pd.DataFrame()
        
        comparison_data = []
        for run in runs:
            ui_sel = run['ui_selections']
            ds_stats = run['dataset_stats']
            test_metrics = run['training_results']['test_metrics']
            
            row = {
                'run_id': run['run_id'],
                'timestamp': run['timestamp'],
                'aoi_mode': ui_sel['aoi_mode'],
                'window_s': ui_sel['window_s'],
                'run_note': ui_sel.get('run_note', ''),
                'n_trials': ds_stats['n_trials'],
                'best_epoch': run['training_results']['best_epoch'],
                'test_accuracy': test_metrics['accuracy'],
                'test_f1_macro': test_metrics['f1_macro'],
                'test_f1_weighted': test_metrics['f1_weighted'],
                'test_precision_macro': test_metrics['precision_macro'],
                'test_recall_macro': test_metrics['recall_macro'],
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric descending
        if primary_metric.startswith('test_'):
            sort_col = primary_metric
        else:
            sort_col = f'test_{primary_metric}'
        
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False)
        
        return df
