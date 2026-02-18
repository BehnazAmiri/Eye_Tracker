"""
Threshold Comparison Manager
Stores and compares results from multiple threshold runs
"""

import json
import os
from datetime import datetime
from pathlib import Path


class ThresholdComparison:
    """Manages storage and comparison of results across different threshold values."""
    
    def __init__(self, results_dir='results/reports'):
        self.results_dir = Path(results_dir)
        self.comparison_file = self.results_dir / 'threshold_comparison.json'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.data = self._load_existing()
    
    def _load_existing(self):
        """Load existing comparison data."""
        if self.comparison_file.exists():
            with open(self.comparison_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'runs': [], 'metadata': {}}
    
    def add_run(self, config, stage1_stats, stage2_stats, stage3_stats):
        """
        Add a new run with its threshold configuration and results.
        
        Args:
            config: ConfigParser object with threshold settings
            stage1_stats: Dictionary with Stage 1 statistics
            stage2_stats: Dictionary with Stage 2 statistics
            stage3_stats: Dictionary with Stage 3 statistics
        """
        # Extract threshold values
        thresholds = {
            'stage1_invalid_pct': config.getfloat('Analysis', 'stage1_invalid_pct_threshold', fallback=0.30),
            'stage1_participant_exclusion': config.getfloat('Analysis', 'stage1_participant_exclusion_threshold', fallback=0.50),
            'ta_window_ms': config.getint('STAGE2_TA', 'ta_window_ms', fallback=1000),
            'ta_coverage': config.getfloat('STAGE2_TA', 'ta_answer_coverage_threshold', fallback=0.85),
            'stage3_percentile': config.getint('Analysis', 'stage3_threshold_percentile', fallback=25)
        }
        
        # Create run entry
        run = {
            'timestamp': datetime.now().isoformat(),
            'thresholds': thresholds,
            'stage1': stage1_stats,
            'stage2': stage2_stats,
            'stage3': stage3_stats
        }
        
        self.data['runs'].append(run)
        self._save()
        
        return len(self.data['runs']) - 1  # Return index of new run
    
    def _save(self):
        """Save comparison data to file."""
        import numpy as np
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        converted_data = convert_types(self.data)
        
        with open(self.comparison_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    def get_all_runs(self):
        """Get all stored runs."""
        return self.data['runs']
    
    def clear_all(self):
        """Clear all stored runs."""
        self.data = {'runs': [], 'metadata': {}}
        self._save()
    
    def get_comparison_table_data(self):
        """
        Generate data for unified comparison table.
        Returns dictionary with single comprehensive table.
        """
        if not self.data['runs']:
            return None
        
        runs = self.data['runs']
        
        # Unified comparison table
        comparison_data = {
            'headers': ['Metric'] + [f"Run {i+1}\n({r['timestamp'][:10]})" for i, r in enumerate(runs)],
            'rows': []
        }
        
        # All metrics in one table with section separators
        all_metrics = [
            # Thresholds section
            ('🔧 THRESHOLDS', None, 'section'),
            ('1. Trial-Level Exclusion', lambda r: f"{r['thresholds']['stage1_invalid_pct']:.0%} invalid", 'threshold'),
            ('2. Participant-Level Exclusion', lambda r: f"{r['thresholds']['stage1_participant_exclusion']:.0%} excluded trials", 'threshold'),
            ('3. Time Window', lambda r: f"{r['thresholds']['ta_window_ms']}ms", 'threshold'),
            ('4. Validity Threshold', lambda r: f"{r['thresholds']['ta_coverage']:.0%} of samples valid", 'threshold'),
            ('5. Percentile Threshold', lambda r: f"P{r['thresholds']['stage3_percentile']}", 'threshold'),
            
            # Stage 1 results
            ('', None, 'separator'),
            ('📊 STAGE 1: Quality Filtering', None, 'section'),
            ('Total Trials', lambda r: f"{r['stage1']['total_trials']}", 'data'),
            ('Trial-Level Excluded', lambda r: f"{r['stage1']['excluded_trials_trial_level']} ({r['stage1']['excluded_trials_trial_level_pct']:.1f}%)", 'data'),
            ('Participant-Level Excluded', lambda r: f"{r['stage1']['excluded_trials_participant_level']} ({r['stage1']['excluded_trials_participant_level_pct']:.1f}%)", 'data'),
            ('Total Excluded', lambda r: f"{r['stage1']['excluded_trials']} ({r['stage1']['excluded_trials_pct']:.1f}%)", 'data'),
            ('Kept Trials', lambda r: f"{r['stage1']['kept_trials']} ({r['stage1']['kept_trials_pct']:.1f}%)", 'data'),
            ('Kept Samples', lambda r: f"{r['stage1']['kept_samples']:,}", 'data'),
            
            # Stage 2 results
            ('', None, 'separator'),
            ('📊 STAGE 2: ta Detection', None, 'section'),
            ('Total Trials', lambda r: f"{r['stage2']['total_trials']}", 'data'),
            ('Trials with ta', lambda r: f"{r['stage2']['trials_with_ta']} ({r['stage2']['ta_detection_rate']:.1f}%)", 'data'),
            ('Trials without ta', lambda r: f"{r['stage2']['trials_without_ta']} ({100-r['stage2']['ta_detection_rate']:.1f}%)", 'data'),
            
            # Stage 3 results
            ('', None, 'separator'),
            ('📊 STAGE 3: Randomness Labeling', None, 'section'),
            ('Total Trials', lambda r: f"{r['stage3']['total_trials']}", 'data'),
            ('Total Valid', lambda r: f"{r['stage3']['total_valid']}", 'data'),
            ('NOT_RANDOM', lambda r: f"{r['stage3']['not_random']} ({r['stage3']['not_random_pct']:.1f}% of valid)", 'data'),
            ('RANDOM', lambda r: f"{r['stage3']['random']} ({r['stage3']['random_pct']:.1f}% of valid)", 'data'),
            ('Invalid', lambda r: f"{r['stage3']['invalid']} ({r['stage3']['invalid_pct']:.1f}% of all)", 'data'),
        ]
        
        for metric_info in all_metrics:
            metric_name = metric_info[0]
            extractor = metric_info[1]
            row_type = metric_info[2]
            
            row = {'name': metric_name, 'type': row_type, 'values': []}
            
            if extractor:
                for run in runs:
                    try:
                        row['values'].append(extractor(run))
                    except (KeyError, TypeError):
                        row['values'].append('N/A')
            else:
                row['values'] = ['' for _ in runs]
            
            comparison_data['rows'].append(row)
        
        return comparison_data
