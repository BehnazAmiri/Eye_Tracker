"""Advanced HTML Report System with Interactive Dashboard."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class AdvancedReportGenerator:
    """Generate professional interactive HTML reports with dashboard."""
    
    def __init__(self, reports_dir: Path):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_dashboard(self):
        """Generate main dashboard with all runs comparison."""
        all_runs = self._load_all_runs()
        
        if not all_runs:
            print("[WARN] No runs found for dashboard")
            return
        
        html = self._build_dashboard_html(all_runs)
        dashboard_path = self.reports_dir / "index.html"
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"[OK] Dashboard generated: {dashboard_path}")
        print(f"  Total runs: {len(all_runs)}")
        
    def generate_detailed_report(self, run_data: Dict, output_path: Path):
        """Generate detailed report for a single run with all visualizations."""
        html = self._build_detailed_html(run_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"[OK] Detailed report: {output_path.name}")
    
    def delete_run(self, json_filename: str, html_filename: str):
        """
        Delete a run's JSON and HTML files, then regenerate dashboard.
        
        Args:
            json_filename: Name of JSON file to delete (e.g., 'mlp_20260213_010901.json')
            html_filename: Name of HTML file to delete (e.g., 'mlp_20260213_010901.html')
        """
        try:
            json_path = self.reports_dir / json_filename
            html_path = self.reports_dir / html_filename
            
            # Delete files
            deleted_files = []
            if json_path.exists():
                json_path.unlink()
                deleted_files.append(json_filename)
                print(f"[OK] Deleted: {json_filename}")
            
            if html_path.exists():
                html_path.unlink()
                deleted_files.append(html_filename)
                print(f"[OK] Deleted: {html_filename}")
            
            if not deleted_files:
                print(f"[WARN] No files found to delete")
                return False
            
            # Regenerate dashboard with updated run list
            print("Regenerating dashboard...")
            self.generate_dashboard()
            
            print(f"[OK] Run deleted successfully: {', '.join(deleted_files)}")
            return True
            
        except Exception as e:
            print(f"[FAIL] Error deleting run: {e}")
            return False
        
    def _load_all_runs(self) -> List[Dict]:
        """Load all JSON reports."""
        runs = []
        for json_file in self.reports_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    data['json_file'] = json_file.name
                    data['html_file'] = json_file.stem + '.html'
                    
                    # Ensure run_id exists
                    if 'run_id' not in data or not data['run_id']:
                        data['run_id'] = json_file.stem  # Use filename as run_id
                    
                    # Ensure timestamp exists and is valid
                    if 'timestamp' not in data or not data['timestamp']:
                        # Try to extract from run_id (format: model_YYYYMMDD_HHMMSS)
                        import re
                        match = re.search(r'(\d{8}_\d{6})', data.get('run_id', ''))
                        if match:
                            data['timestamp'] = match.group(1)
                        else:
                            # Fallback to file modification time
                            from datetime import datetime
                            mtime = json_file.stat().st_mtime
                            data['timestamp'] = datetime.fromtimestamp(mtime).strftime('%Y%m%d_%H%M%S')
                            
                    runs.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {json_file.name}: {e}")
        
        # Sort by timestamp (newest first)
        # Convert to string to ensure consistent comparison
        runs.sort(key=lambda x: str(x.get('timestamp', '')), reverse=True)
        return runs
    
    def _build_dashboard_html(self, runs: List[Dict]) -> str:
        """Build interactive dashboard HTML."""
        
        # Helper to get metrics from either flat (LOPO) or nested (Simple) structure
        def get_val(r, k):
            m = r.get('metrics', {})
            return m.get('test', {}).get(k, 0) if 'test' in m else m.get(k, 0)
        
        # Find best run for each metric
        best_acc = max(runs, key=lambda x: get_val(x, 'accuracy'))
        best_auc = max(runs, key=lambda x: get_val(x, 'roc_auc'))
        best_f1 = max(runs, key=lambda x: get_val(x, 'f1'))
        
        # Build table rows
        table_rows = []
        total_runs = len(runs)
        for i, run in enumerate(runs):
            # Calculate chronological index (Oldest = #1, Newest = #Total)
            # Since runs are sorted Newest->Oldest, we reverse the index
            idx = total_runs - i
            
            # Robust metric extraction
            raw_metrics = run.get('metrics', {})
            metrics = raw_metrics.get('test', raw_metrics) if 'test' in raw_metrics else raw_metrics
            
            data_config = run.get('data_config', {})
            
            acc = metrics.get('accuracy', 0) * 100
            auc = metrics.get('roc_auc', 0)
            f1 = metrics.get('f1', 0)
            
            # Highlight best
            is_best_acc = (run['run_id'] == best_acc['run_id'])
            is_best_auc = (run['run_id'] == best_auc['run_id'])
            is_best_f1 = (run['run_id'] == best_f1['run_id'])
            
            row_class = 'best-run' if (is_best_acc or is_best_auc or is_best_f1) else ''
            
            # Extract filter info
            parts = data_config.get('parts_filter', [])
            parts_str = ', '.join(parts) if parts else 'All'
            if len(parts_str) > 30:
                parts_str = parts_str[:27] + '...'
            
            aoi = data_config.get('aoi_filter', [])
            aoi_str = ', '.join(aoi) if aoi else 'All'
            if len(aoi_str) > 20:
                aoi_str = aoi_str[:17] + '...'
            
            # Features info
            feat_cols = data_config.get('feature_columns', [])
            
            # Smart detection of feature set
            is_selected_set = feat_cols and ('LPD' in feat_cols or 'LPD_mean' in feat_cols) and ('RPD' in feat_cols or 'RPD_mean' in feat_cols)
            is_default_set = feat_cols and ('LPOGX' in feat_cols or 'LPOGX_mean' in feat_cols) and ('RPOGX' in feat_cols or 'RPOGX_mean' in feat_cols)
            is_all_features = len(feat_cols) > 20 if feat_cols else False  # 41 features = all
            
            count = len(feat_cols) if feat_cols else data_config.get('n_feature_columns', '?')
            
            # Check count FIRST before checking content
            if is_all_features:
                 feat_display = "All Features"
                 if feat_cols:
                     feat_tooltip = f"All Features ({len(feat_cols)}): " + ', '.join(feat_cols[:10]) + ('...' if len(feat_cols) > 10 else '')
                 else:
                     feat_tooltip = "All Features (Default)"
            elif is_selected_set:
                 if count == 65:
                     feat_display = "Selected (13 Aggregated)"
                 elif count == 13:
                     feat_display = "Selected (13 Raw)"
                 else:
                     feat_display = f"Selected ({count})"
                 feat_tooltip = "Selected High-Quality Features: " + ', '.join(feat_cols)
            elif is_default_set:
                 feat_display = "All Features"
                 feat_tooltip = ', '.join(feat_cols)
            elif feat_cols and len(feat_cols) == 13:
                 # Fallback for when detection fails but count is 13
                 feat_display = "All Features" 
                 feat_tooltip = ', '.join(feat_cols)
            else:
                 count = len(feat_cols) if feat_cols else data_config.get('n_feature_columns', '?')
                 feat_display = f"{count} Features"
                 feat_tooltip = ', '.join(feat_cols) if feat_cols else 'All Features'

            # Get evaluation method
            eval_method = data_config.get('evaluation_method', 'N/A').upper()
            
            table_rows.append(f"""
                <tr class="{row_class}" onclick="window.location='{run['html_file']}'" data-run-id="{run['run_id']}" data-json-file="{run['json_file']}" data-html-file="{run['html_file']}">
                    <td><strong>#{idx}</strong></td>
                    <td><span class="badge badge-{run['model_type']}">{run['model_type'].upper()}</span></td>
                    <td>{eval_method}</td>
                    <td>{('Full trial' if data_config.get('time_window_s') == 'full' else f"{data_config.get('time_window_s', 'N/A')}s")}</td>
                    <td class="{'highlight-best' if is_best_acc else ''}">{acc:.2f}%</td>
                    <td class="{'highlight-best' if is_best_auc else ''}">{auc:.4f}</td>
                    <td class="{'highlight-best' if is_best_f1 else ''}">{f1:.4f}</td>
                    <td>{data_config.get('n_train', 'N/A')}/{data_config.get('n_test', 'N/A')}</td>
                    <td title="{feat_tooltip}">{feat_display}</td>
                    <td title="{', '.join(parts)}">{parts_str}</td>
                    <td title="{', '.join(aoi)}">{aoi_str}</td>
                    <td><button class="delete-btn" onclick="event.stopPropagation(); deleteRun('{run['run_id']}', '{run['json_file']}', '{run['html_file']}')">Delete</button></td>
                </tr>
            """)
        
        # Prepare header stats
        b_acc_val = get_val(best_acc, 'accuracy') * 100
        b_auc_val = get_val(best_auc, 'roc_auc')
        b_f1_val = get_val(best_f1, 'f1')

        # Prepare dynamic filter lists
        # 1. Models
        unique_models = sorted(list(set(r['model_type'] for r in runs)))
        model_checklists = ""
        for m in unique_models:
             m_val = m.lower()
             m_label = m.upper()
             if m_val == 'rf': m_label = 'Random Forest'
             if m_val == 'logreg': m_label = 'Logistic Regression'
             if m_val == 'bilstm': m_label = 'BiLSTM'
             if m_val == 'cnn1d': m_label = 'CNN-1D (Alt)'
             model_checklists += f'<label class="checkbox-item"><input type="checkbox" value="{m_val}" onchange="filterTable()"> {m_label}</label>\n'

        # 2. AOIs - FIXED LIST from GUI
        aoi_checklists = ""
        # Option 1: Answer Area
        aoi_checklists += '<label class="checkbox-item"><input type="checkbox" value="answer_area" onchange="filterTable()"> Answer Area</label>\n'
        # Option 2: All AOIs
        aoi_checklists += '<label class="checkbox-item"><input type="checkbox" value="all_aois" onchange="filterTable()"> All AOIs</label>\n'
        # Add dynamic ones if they exist and aren't covered? No, user requested specific filters.
        # But if we have runs with other AOIs, they won't be filterable?
        # The user said "Must be these". So we stick to these.

        # 3. Parts - FIXED LIST from GUI
        parts_checklists = ""
        # Option 1: Timer Trials
        parts_checklists += '<label class="checkbox-item"><input type="checkbox" value="timer_trials" onchange="filterTable()"> Timer Trials</label>\n'
        # Option 2: No-Timer + Timer-No-Correct
        parts_checklists += '<label class="checkbox-item"><input type="checkbox" value="notimer_incorrect" onchange="filterTable()"> No-Timer + Timer-No-Correct</label>\n'
        # Option 3: All Parts
        parts_checklists += '<label class="checkbox-item"><input type="checkbox" value="all_parts" onchange="filterTable()"> All Parts</label>\n'
        
        # 4. Evaluation Methods
        eval_checklists = ""
        eval_checklists += '<label class="checkbox-item"><input type="checkbox" value="lopo" onchange="filterTable()"> LOPO</label>\n'
        eval_checklists += '<label class="checkbox-item"><input type="checkbox" value="simple" onchange="filterTable()"> Simple</label>\n'
        
        # 5. Time Windows - Dynamic from runs
        unique_time_windows = []
        for r in runs:
            tw = r.get('data_config', {}).get('time_window_s')
            if tw is not None and tw not in unique_time_windows:
                unique_time_windows.append(tw)
        
        # Sort: numbers first (ascending), then 'full' at end
        def sort_time_window(tw):
            if tw == 'full':
                return (1, 9999)  # Sort 'full' at end
            elif isinstance(tw, (int, float)):
                return (0, tw)
            else:
                return (0, 0)
        
        unique_time_windows.sort(key=sort_time_window)
        time_checklists = ""
        for tw in unique_time_windows:
            if tw == 'full':
                tw_val = 'full'
                tw_label = 'Full trial'
            else:
                tw_val = str(tw)
                tw_label = f'{tw}s'
            time_checklists += f'<label class="checkbox-item"><input type="checkbox" value="{tw_val}" onchange="filterTable()"> {tw_label}</label>\n'


        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Learning Pipeline - Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f1f5f9;
            color: #334155;
            padding: 20px;
            min-height: 100vh;
            font-size: 0.875rem; 
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            overflow: hidden;
        }}
        
        .header {{
            background: #ffffff;
            color: #0f172a;
            padding: 24px 32px;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .header h1 {{
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: #0f172a;
            margin-bottom: 4px;
        }}
        
        .header p {{
            font-size: 0.875rem;
            color: #64748b;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 24px 32px;
            background: #f8fafc;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            text-align: left;
        }}
        
        .stat-card h3 {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #3b82f6;
            margin-bottom: 4px;
            letter-spacing: -0.025em;
        }}
        
        .stat-card p {{
            color: #64748b;
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .content {{
            padding: 32px;
        }}
        
        .filters {{
            display: flex;
            gap: 16px;
            margin-bottom: 24px;
            padding: 20px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            flex-wrap: wrap;
        }}
        
        .filter-group {{
            flex: 1;
            min-width: 180px;
        }}
        
        .filter-group label {{
            display: block;
            font-weight: 600;
            margin-bottom: 6px;
            color: #475569;
            font-size: 0.75rem;
        }}
        
        .filter-group input,
        .filter-group select {{
            width: 100%;
            padding: 8px 10px;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            font-size: 0.85rem;
            color: #334155;
            background-color: #f8fafc;
            transition: all 0.2s;
            font-family: inherit;
        }}

        /* Custom Multiselect Checkbox Style */
        .filter-dropdown {{
            position: relative;
        }}
        
        .filter-btn {{
            width: 100%;
            padding: 8px 10px;
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            cursor: pointer;
            text-align: left;
            font-size: 0.85rem;
            color: #334155;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .filter-btn:after {{
            content: '?';
            font-size: 0.6rem;
            color: #94a3b8;
        }}
        
        .dropdown-content {{
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background-color: white;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            z-index: 50;
            max-height: 300px;
            overflow-y: auto;
            padding: 8px;
            margin-top: 4px;
        }}
        
        .dropdown-content.show {{
            display: block;
        }}
        
        .checkbox-item {{
            display: flex;
            align-items: center;
            padding: 6px 8px;
            cursor: pointer;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #475569;
        }}
        
        .checkbox-item:hover {{
            background-color: #f1f5f9;
        }}
        
        .checkbox-item input {{
            width: auto;
            margin-right: 10px;
        }}
        
        .filter-group input:focus,
        .filter-group select:focus {{
            outline: none;
            border-color: #3b82f6;
            ring: 2px solid #3b82f633;
            background-color: #fff;
        }}
        
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 16px;
            font-size: 0.85rem;
        }}
        
        thead {{
            background: #f8fafc;
        }}
        
        th {{
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #475569;
            border-bottom: 1px solid #e2e8f0;
            border-top: 1px solid #e2e8f0;
            cursor: pointer;
            user-select: none;
            text-transform: uppercase;
            font-size: 0.70rem;
            letter-spacing: 0.05em;
        }}
        
        th:hover {{
            background: #f1f5f9;
            color: #1e293b;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
            color: #334155;
        }}
        
        tr {{
            transition: background-color 0.15s;
        }}
        
        tr:hover {{
            background: #f8fafc;
        }}
        
        .best-run {{
            background: #f0fdf4 !important;
        }}
        
        .best-run td {{
            color: #166534;
        }}

        .best-run td strong {{
            color: #166534;
        }}
        
        .highlight-best {{
            color: #15803d;
            font-weight: 700;
            background: #dcfce7;
            padding: 2px 8px;
            border-radius: 4px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .badge-mlp {{ background: #e0f2fe; color: #075985; }}
        .badge-cnn {{ background: #ffedd5; color: #9a3412; }}
        .badge-cnn1d {{ background: #fef3c7; color: #92400e; }}
        .badge-lstm {{ background: #fae8ff; color: #86198f; }}
        .badge-bilstm {{ background: #ede9fe; color: #5b21b6; }}
        .badge-hybrid {{ background: #f3e8ff; color: #6b21a8; }}
        .badge-transformer {{ background: #ccfbf1; color: #115e59; }}
        .badge-rf {{ background: #dcfce7; color: #166534; }}
        .badge-svm {{ background: #f1f5f9; color: #475569; }}
        .badge-logreg {{ background: #e0e7ff; color: #3730a3; }}
        .badge-xgboost {{ background: #fee2e2; color: #991b1b; }}
        
        .active-filters-box {{
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 20px;
            display: none;
        }}
        
        .active-filters-box.show {{
            display: block;
        }}
        
        .active-filters-title {{
            font-weight: 600;
            color: #1e40af;
            margin-bottom: 8px;
            font-size: 0.9em;
        }}
        
        .active-filter-tag {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #dbeafe;
            color: #1e40af;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            margin: 4px 4px 4px 0;
            border: 1px solid #93c5fd;
        }}
        
        .remove-filter-btn {{
            background: transparent;
            border: none;
            color: #1e40af;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: bold;
            padding: 0;
            margin: 0;
            line-height: 1;
            transition: all 0.2s;
        }}
        
        .remove-filter-btn:hover {{
            color: #dc2626;
            transform: scale(1.2);
        }}
        
        .clear-filters-btn {{
            background: #ef4444;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 600;
            transition: all 0.2s;
            margin-left: auto;
        }}
        
        .clear-filters-btn:hover:not(:disabled) {{
            background: #dc2626;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(239, 68, 68, 0.3);
        }}
        
        .clear-filters-btn:disabled {{
            background: #9ca3af;
            cursor: not-allowed;
            opacity: 0.5;
        }}
        
        .delete-btn {{
            background: #fee2e2;
            color: #dc2626;
            border: 1px solid #fecaca;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }}
        
        .delete-btn:hover {{
            background: #dc2626;
            color: white;
            border-color: #dc2626;
        }}
        
        a {{
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .no-data {{
            text-align: center;
            padding: 60px;
            color: #a0aec0;
            font-size: 1.2em;
        }}
        
        @media (max-width: 768px) {{
            .filters {{
                flex-direction: column;
            }}
            
            table {{
                font-size: 0.9em;
            }}
            
            th, td {{
                padding: 10px 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>Deep Learning Pipeline Dashboard</h1>
                <p>Comprehensive Model Performance Tracking</p>
            </div>
            <div class="generated-time">
                Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>{len(runs)}</h3>
                <p>Total Runs</p>
            </div>
            <div class="stat-card">
                <h3>{b_acc_val:.2f}%</h3>
                <p>Best Accuracy</p>
            </div>
            <div class="stat-card">
                <h3>{b_auc_val:.4f}</h3>
                <p>Best ROC-AUC</p>
            </div>
            <div class="stat-card">
                <h3>{b_f1_val:.4f}</h3>
                <p>Best F1-Score</p>
            </div>
            <div class="stat-card">
                <h3>{len(set(r['model_type'] for r in runs))}</h3>
                <p>Models Tested</p>
            </div>
        </div>
        
        <div class="content">
            <!-- Active Filters Display -->
            <div id="activeFiltersBox" class="active-filters-box">
                <div class="active-filters-title">Active Filters:</div>
                <div id="activeFiltersList"></div>
            </div>
            
            <div class="filters" style="display: flex; align-items: flex-end; gap: 15px;">
                <div class="filter-group">
                    <label>Parts Filter</label>
                    <div class="filter-dropdown">
                        <button class="filter-btn" onclick="toggleDropdown('partsDropdown')">Select Parts...</button>
                        <div id="partsDropdown" class="dropdown-content">
                            {parts_checklists}
                        </div>
                    </div>
                </div>
                <div class="filter-group">
                    <label>AOI Filter</label>
                    <div class="filter-dropdown">
                        <button class="filter-btn" onclick="toggleDropdown('aoiDropdown')">Select AOIs...</button>
                        <div id="aoiDropdown" class="dropdown-content">
                            {aoi_checklists}
                        </div>
                    </div>
                </div>
                <div class="filter-group">
                    <label>Model Type</label>
                    <div class="filter-dropdown">
                        <button class="filter-btn" onclick="toggleDropdown('modelDropdown')">Select Models...</button>
                        <div id="modelDropdown" class="dropdown-content">
                            {model_checklists}
                        </div>
                    </div>
                </div>
                <div class="filter-group">
                    <label>Evaluation Method</label>
                    <div class="filter-dropdown">
                        <button class="filter-btn" onclick="toggleDropdown('evalDropdown')">Select Method...</button>
                        <div id="evalDropdown" class="dropdown-content">
                            {eval_checklists}
                        </div>
                    </div>
                </div>
                <div class="filter-group">
                    <label>Time Window</label>
                    <div class="filter-dropdown">
                        <button class="filter-btn" onclick="toggleDropdown('timeDropdown')">Select Time Windows...</button>
                        <div id="timeDropdown" class="dropdown-content">
                            {time_checklists}
                        </div>
                    </div>
                </div>
                <div class="filter-group">
                    <label>Features</label>
                    <select id="featureFilter">
                        <option value="">All Feature Sets</option>
                        <option value="selected">Selected Set (13)</option>
                        <option value="default">All Features</option>
                    </select>
                </div>
                <div class="filter-group" style="margin-bottom: 0;">
                    <button id="clearFiltersBtn" class="clear-filters-btn" onclick="clearAllFilters()" disabled>
                        Clear All Filters
                    </button>
                </div>
            </div>
            
            <table id="runsTable">
                <thead>
                    <tr>
                        <th onclick="sortTable(0)" style="width: 50px;">#</th>
                        <th onclick="sortTable(1)">Model</th>
                        <th onclick="sortTable(2)">Eval Method</th>
                        <th onclick="sortTable(3)">Time Win (s)</th>
                        <th onclick="sortTable(4)">Accuracy (%)</th>
                        <th onclick="sortTable(5)">ROC-AUC</th>
                        <th onclick="sortTable(6)">F1-Score</th>
                        <th onclick="sortTable(7)">Train/Test</th>
                        <th onclick="sortTable(8)">Features</th>
                        <th onclick="sortTable(9)">Parts Filter</th>
                        <th onclick="sortTable(10)">AOI Filter</th>
                        <th style="width: 80px;">Actions</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                    {''.join(table_rows)}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Dropdown Toggle Logic
        function toggleDropdown(id) {{
            var dropdown = document.getElementById(id);
            dropdown.classList.toggle("show");
        }}

        // Close dropdowns if clicked outside
        window.onclick = function(event) {{
            if (!event.target.matches('.filter-btn')) {{
                var dropdowns = document.getElementsByClassName("dropdown-content");
                for (var i = 0; i < dropdowns.length; i++) {{
                    var openDropdown = dropdowns[i];
                    if (openDropdown.classList.contains('show')) {{
                        openDropdown.classList.remove('show');
                    }}
                }}
            }}
        }}

        // Helper to get values from checked boxes
        function getCheckedValues(dropdownId) {{
            const container = document.getElementById(dropdownId);
            if (!container) return [];
            const checkboxes = container.querySelectorAll('input[type="checkbox"]:checked');
            return Array.from(checkboxes).map(cb => cb.value.toLowerCase());
        }}

        // Copy delete command to clipboard
        function copyDeleteCommand() {{
            const textarea = document.getElementById('deleteCommand');
            if (textarea) {{
                textarea.select();
                textarea.setSelectionRange(0, 99999); // For mobile devices
                
                // Try modern clipboard API first
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(textarea.value).then(function() {{
                        alert('[OK] Command copied to clipboard!');
                    }}).catch(function(err) {{
                        // Fallback to execCommand
                        document.execCommand('copy');
                        alert('[OK] Command copied to clipboard!');
                    }});
                }} else {{
                    // Fallback to execCommand for older browsers
                    document.execCommand('copy');
                    alert('[OK] Command copied to clipboard!');
                }}
            }}
        }}

        // Delete Run Functionality
        function deleteRun(runId, jsonFile, htmlFile) {{
            // Create a modal/dialog to show the command
            if (confirm('[WARN]? DELETE RUN\\n\\nClick OK to see the delete command.')) {{
                // Prepare the command
                const command = `cd D:\\\\MasterThesis\\\\MasterThesis\\\\DeepLearning
python -c "from src.reporting.html_report import AdvancedReportGenerator; from pathlib import Path; gen = AdvancedReportGenerator(Path('outputs/reports')); gen.delete_run('${{jsonFile}}', '${{htmlFile}}')"`;
                
                // Create modal
                const modal = document.createElement('div');
                modal.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0,0,0,0.7);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 9999;
                `;
                
                const modalContent = document.createElement('div');
                modalContent.style.cssText = `
                    background: white;
                    padding: 30px;
                    border-radius: 12px;
                    max-width: 700px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                `;
                
                // Create textarea to hold command (hidden)
                const commandTextArea = document.createElement('textarea');
                commandTextArea.id = 'deleteCommand';
                commandTextArea.value = command;
                commandTextArea.style.position = 'absolute';
                commandTextArea.style.left = '-9999px';
                
                modalContent.innerHTML = `
                    <h2 style="color: #dc2626; margin-bottom: 20px;">
                        Delete Run
                    </h2>
                    <p style="margin-bottom: 15px; color: #475569;">
                        <strong>Run ID:</strong> ${{runId}}<br>
                        <strong>Files to delete:</strong><br>
                        ? ${{htmlFile}}<br>
                        ? ${{jsonFile}}
                    </p>
                    <div style="background: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; margin: 20px 0;">
                        <p style="margin-bottom: 10px; font-weight: 600; color: #1e293b;">Copy and run this command in PowerShell:</p>
                        <pre id="commandPre" style="background: #1e293b; color: #e2e8f0; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85em; font-family: 'Courier New', monospace; cursor: pointer;" onclick="copyDeleteCommand()">${{command}}</pre>
                    </div>
                    <p style="color: #64748b; font-size: 0.9em; margin: 15px 0;">
                        ? After running the command, refresh this page to see the updated dashboard.
                    </p>
                    <button onclick="this.parentElement.parentElement.remove()" style="
                        background: #dc2626;
                        color: white;
                        border: none;
                        padding: 10px 24px;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 1em;
                        font-weight: 600;
                        margin-top: 10px;
                    ">Close</button>
                    <button id="copyBtn" onclick="copyDeleteCommand()" style="
                        background: #3b82f6;
                        color: white;
                        border: none;
                        padding: 10px 24px;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 1em;
                        font-weight: 600;
                        margin-top: 10px;
                        margin-left: 10px;
                    ">[NOTE] Copy Command</button>
                `;
                
                modalContent.appendChild(commandTextArea);
                
                modal.appendChild(modalContent);
                document.body.appendChild(modal);
                
                // Close modal when clicking outside
                modal.addEventListener('click', function(e) {{
                    if (e.target === modal) {{
                        modal.remove();
                    }}
                }});
            }}
        }}
        
        // Re-number rows after deletion (not needed for manual delete, but kept for future use)
        function renumberRows() {{
            const table = document.getElementById('tableBody');
            const rows = table.getElementsByTagName('tr');
            const totalRows = rows.length;
            
            for (let i = 0; i < totalRows; i++) {{
                const row = rows[i];
                const indexCell = row.cells[0];
                // Reverse index (oldest = 1, newest = totalRows)
                indexCell.innerHTML = '<strong>#' + (totalRows - i) + '</strong>';
            }}
        }}
        
        // Update stats after deletion (not needed for manual delete, but kept for future use)
        function updateStats() {{
            const table = document.getElementById('tableBody');
            const totalRuns = table.getElementsByTagName('tr').length;
            
            // Update total runs display (if exists in stats section)
            const statCards = document.querySelectorAll('.stat-card h3');
            if (statCards.length > 0) {{
                statCards[0].textContent = totalRuns;
            }}
        }}

        // Filter functionality
        document.getElementById('featureFilter').addEventListener('change', filterTable);
        
        function updateActiveFiltersDisplay(partsValues, aoiValues, modelValues, evalValues, timeValues, featureValue) {{
            const box = document.getElementById('activeFiltersBox');
            const list = document.getElementById('activeFiltersList');
            const clearBtn = document.getElementById('clearFiltersBtn');
            
            let filterTags = [];
            
            // Model filters
            if (modelValues.length > 0) {{
                modelValues.forEach(val => {{
                    let label = val.toUpperCase();
                    if (val === 'rf') label = 'Random Forest';
                    if (val === 'logreg') label = 'Logistic Regression';
                    if (val === 'bilstm') label = 'BiLSTM';
                    if (val === 'cnn1d') label = 'CNN-1D (Alt)';
                    filterTags.push(`<span class="active-filter-tag">Model: ${{label}}<button class="remove-filter-btn" onclick="removeFilter('model', '${{val}}')">?</button></span>`);
                }});
            }}
            
            // Evaluation Method filters
            if (evalValues.length > 0) {{
                evalValues.forEach(val => {{
                    let label = val.toUpperCase();
                    filterTags.push(`<span class="active-filter-tag">Eval: ${{label}}<button class="remove-filter-btn" onclick="removeFilter('eval', '${{val}}')">?</button></span>`);
                }});
            }}
            
            // Parts filters
            if (partsValues.length > 0) {{
                partsValues.forEach(val => {{
                    let label = val;
                    if (val === 'timer_trials') label = 'Timer Trials';
                    if (val === 'notimer_incorrect') label = 'No-Timer + Timer-No-Correct';
                    if (val === 'all_parts') label = 'All Parts';
                    filterTags.push(`<span class="active-filter-tag">Parts: ${{label}}<button class="remove-filter-btn" onclick="removeFilter('parts', '${{val}}')">?</button></span>`);
                }});
            }}
            
            // AOI filters
            if (aoiValues.length > 0) {{
                aoiValues.forEach(val => {{
                    let label = val;
                    if (val === 'answer_area') label = 'Answer Area';
                    if (val === 'all_aois') label = 'All AOIs';
                    filterTags.push(`<span class="active-filter-tag">AOI: ${{label}}<button class="remove-filter-btn" onclick="removeFilter('aoi', '${{val}}')">?</button></span>`);
                }});
            }}
            
            // Time Window filters
            if (timeValues.length > 0) {{
                timeValues.forEach(val => {{
                    let label = val === 'full' ? 'Full trial' : val + 's';
                    filterTags.push(`<span class="active-filter-tag">Time: ${{label}}<button class="remove-filter-btn" onclick="removeFilter('time', '${{val}}')">?</button></span>`);
                }});
            }}
            
            // Feature filter
            if (featureValue) {{
                let label = featureValue;
                if (featureValue === 'selected') label = 'Selected Set (13)';
                if (featureValue === 'default') label = 'All Features';
                filterTags.push(`<span class="active-filter-tag">Features: ${{label}}<button class="remove-filter-btn" onclick="removeFilter('features', '${{featureValue}}')">?</button></span>`);
            }}
            
            // Update display
            if (filterTags.length > 0) {{
                list.innerHTML = filterTags.join('');
                box.classList.add('show');
                clearBtn.disabled = false;
            }} else {{
                box.classList.remove('show');
                clearBtn.disabled = true;
            }}
        }}
        
        function removeFilter(filterType, filterValue) {{
            // Remove specific filter based on type
            if (filterType === 'model') {{
                // Find and uncheck the corresponding checkbox in modelDropdown
                const checkboxes = document.querySelectorAll('#modelDropdown input[type="checkbox"]');
                checkboxes.forEach(cb => {{
                    if (cb.value === filterValue) {{
                        cb.checked = false;
                    }}
                }});
            }} else if (filterType === 'eval') {{
                // Find and uncheck the corresponding checkbox in evalDropdown
                const checkboxes = document.querySelectorAll('#evalDropdown input[type="checkbox"]');
                checkboxes.forEach(cb => {{
                    if (cb.value === filterValue) {{
                        cb.checked = false;
                    }}
                }});
            }} else if (filterType === 'parts') {{
                // Find and uncheck the corresponding checkbox in partsDropdown
                const checkboxes = document.querySelectorAll('#partsDropdown input[type="checkbox"]');
                checkboxes.forEach(cb => {{
                    if (cb.value === filterValue) {{
                        cb.checked = false;
                    }}
                }});
            }} else if (filterType === 'aoi') {{
                // Find and uncheck the corresponding checkbox in aoiDropdown
                const checkboxes = document.querySelectorAll('#aoiDropdown input[type="checkbox"]');
                checkboxes.forEach(cb => {{
                    if (cb.value === filterValue) {{
                        cb.checked = false;
                    }}
                }});
            }} else if (filterType === 'time') {{
                // Find and uncheck the corresponding checkbox in timeDropdown
                const checkboxes = document.querySelectorAll('#timeDropdown input[type="checkbox"]');
                checkboxes.forEach(cb => {{
                    if (cb.value === filterValue) {{
                        cb.checked = false;
                    }}
                }});
            }} else if (filterType === 'features') {{
                // Reset feature filter select
                document.getElementById('featureFilter').value = '';
            }}
            
            // Re-filter table
            filterTable();
        }}
        
        function clearAllFilters() {{
            // Uncheck all checkboxes
            document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
            
            // Reset feature filter
            document.getElementById('featureFilter').value = '';
            
            // Re-filter table (which will show all rows)
            filterTable();
        }}
        
        function filterTable() {{
            // Get selected values
            const partsValues = getCheckedValues('partsDropdown');
            const aoiValues = getCheckedValues('aoiDropdown');
            const modelValues = getCheckedValues('modelDropdown');
            const evalValues = getCheckedValues('evalDropdown');
            const timeValues = getCheckedValues('timeDropdown');
            
            const featureValue = document.getElementById('featureFilter').value;
            
            // Update active filters display
            updateActiveFiltersDisplay(partsValues, aoiValues, modelValues, evalValues, timeValues, featureValue);
             
            const table = document.getElementById('tableBody');
            const rows = table.getElementsByTagName('tr');
            
            for (let i = 0; i < rows.length; i++) {{
                const row = rows[i];
                const model = row.cells[1].textContent.toLowerCase();
                const evalMethod = row.cells[2].textContent.toLowerCase(); // Eval Method column
                const timeWindow = row.cells[3].textContent.trim(); // Time Window column
                 // Use title attribute for full text
                const partsText = (row.cells[9].getAttribute('title') || '').toLowerCase();
                const aoiText = (row.cells[10].getAttribute('title') || '').toLowerCase();
                const featuresInfo = row.cells[8].textContent.trim();
                
                let showRow = true;
                
                // Parts Filter (Multi-Select OR Logic)
                if (partsValues.length > 0) {{
                    let partMatch = false;
                    for (const val of partsValues) {{
                        if (val === 'timer_trials') {{
                             // Timer-Correct trials (not including No-Timer trials)
                             if (partsText.includes('timer-correct') && !partsText.includes('no-timer')) {{
                                 partMatch = true;
                             }}
                        }} else if (val === 'notimer_incorrect') {{
                             // No-Timer trials OR Timer-No-Correct trials
                             if (partsText.includes('no-timer') || partsText.includes('timer-no-correct')) {{
                                 partMatch = true;
                             }}
                        }} else if (val === 'all_parts') {{
                             // All parts (empty filter or explicitly "all")
                             if (partsText === '' || partsText === 'all' || partsText.includes('all')) {{
                                 partMatch = true;
                             }}
                        }} else {{
                             // Generic match for any other custom values
                             if (partsText.includes(val)) partMatch = true;
                        }}
                    }}
                    if (!partMatch) showRow = false;
                }}

                // AOI Filter (Multi-Select OR Logic)
                if (showRow && aoiValues.length > 0) {{
                    let aoiMatch = false; 
                    for (const val of aoiValues) {{
                        if (val === 'answer_area') {{
                            if (aoiText.includes('answer_area')) aoiMatch = true;
                        }} else if (val === 'all_aois') {{
                            // All AOIs (empty filter or explicitly "all")
                            if (aoiText === '' || aoiText === 'all' || aoiText.includes('all')) aoiMatch = true;
                        }} else if (val === '[]') {{
                            // Empty AOI filter
                            if (aoiText === '' || aoiText.includes('none')) aoiMatch = true;
                        }} else if (val === 'question') {{
                             if (aoiText.includes('question')) aoiMatch = true;
                        }} else {{
                            // Generic match for any other custom values
                            if (aoiText.includes(val)) aoiMatch = true;
                        }}
                    }}
                    if (!aoiMatch) showRow = false;
                }}
                
                // Model Filter (Multi-Select OR Logic)
                if (showRow && modelValues.length > 0) {{
                    let modelMatch = false;
                    for (const val of modelValues) {{
                        if (model.includes(val)) modelMatch = true;
                    }}
                    if (!modelMatch) showRow = false;
                }}
                
                // Evaluation Method Filter (Multi-Select OR Logic)
                if (showRow && evalValues.length > 0) {{
                    let evalMatch = false;
                    for (const val of evalValues) {{
                        if (evalMethod.includes(val)) evalMatch = true;
                    }}
                    if (!evalMatch) showRow = false;
                }}
                
                // Time Window Filter (Multi-Select OR Logic)
                if (showRow && timeValues.length > 0) {{
                    let timeMatch = false;
                    for (const val of timeValues) {{
                        if (val === 'full') {{
                            if (timeWindow === 'Full trial') timeMatch = true;
                        }} else {{
                            if (timeWindow === val + 's') timeMatch = true;
                        }}
                    }}
                    if (!timeMatch) showRow = false;
                }}

                // Features Filter
                if (showRow && featureValue) {{
                    if (featureValue === 'selected' && !featuresInfo.startsWith('Selected')) {{
                        showRow = false;
                    }} else if (featureValue === 'default' && !featuresInfo.includes('All Features')) {{
                        showRow = false;
                    }}
                }}
                
                row.style.display = showRow ? '' : 'none';
            }}
            
            updateBestRunHighlight();
        }}
        
        function updateBestRunHighlight() {{
            const table = document.getElementById('tableBody');
            const rows = Array.from(table.rows);
            
            // Remove all best-run classes first
            rows.forEach(row => row.classList.remove('best-run'));
            
            // Find visible rows
            const visibleRows = rows.filter(row => row.style.display !== 'none');
            
            if (visibleRows.length === 0) return;
            
            // Find row with highest accuracy (column index 3)
            let bestRow = visibleRows[0];
            let bestAcc = parseFloat(visibleRows[0].cells[3].textContent.replace('%', '')) || 0;
            
            visibleRows.forEach(row => {{
                const acc = parseFloat(row.cells[3].textContent.replace('%', '')) || 0;
                if (acc > bestAcc) {{
                    bestAcc = acc;
                    bestRow = row;
                }}
            }});
            
            // Highlight the best row
            bestRow.classList.add('best-run');
        }}
        
        // Sort functionality
        let sortDirection = {{}};
        
        function sortTable(columnIndex) {{
            const table = document.getElementById('tableBody');
            const rows = Array.from(table.rows);
            
            const direction = sortDirection[columnIndex] || 'asc';
            sortDirection[columnIndex] = direction === 'asc' ? 'desc' : 'asc';
            
            rows.sort((a, b) => {{
                let aValue = a.cells[columnIndex].textContent.trim();
                let bValue = b.cells[columnIndex].textContent.trim();
                
                // Handle run number column
                if (columnIndex === 0) {{
                    aValue = parseInt(aValue.replace('#', '')) || 0;
                    bValue = parseInt(bValue.replace('#', '')) || 0;
                }}
                // Handle Time Win (s)
                else if (columnIndex === 2) {{
                    aValue = parseFloat(aValue.replace('s', '')) || 0;
                    bValue = parseFloat(bValue.replace('s', '')) || 0;
                }}
                // Handle numeric columns (Accuracy, AUC, F1, features)
                else if (columnIndex === 3 || columnIndex === 4 || columnIndex === 5 || columnIndex === 7) {{
                    aValue = parseFloat(aValue.replace('%', '')) || 0;
                    bValue = parseFloat(bValue.replace('%', '')) || 0;
                }}
                
                if (direction === 'asc') {{
                    return aValue > bValue ? 1 : -1;
                }} else {{
                    return aValue < bValue ? 1 : -1;
                }}
            }});
            
            rows.forEach(row => table.appendChild(row));
            
            // Update best-run highlighting after sort
            updateBestRunHighlight();
        }}
        
        // Initialize best-run highlighting on page load
        updateBestRunHighlight();
    </script>
</body>
</html>"""
    
    def _build_detailed_html(self, run_data: Dict) -> str:
        """Build detailed HTML report for a single run."""
        
        # Check evaluation method
        eval_method = run_data.get('data_config', {}).get('evaluation_method', 'simple')
        is_lopo = (eval_method == 'lopo')
        
        # Generate visualizations
        confusion_matrix_img = self._create_confusion_matrix(run_data)
        metrics_comparison_img = self._create_metrics_comparison(run_data)
        predictions_dist_img = self._create_predictions_distribution(run_data)
        roc_curve_img = self._create_roc_curve(run_data)
        class_performance_img = self._create_class_performance(run_data)
        training_history_img = self._create_training_history(run_data)
        
        # For LOPO, metrics are flat; for Simple, they're nested under 'test'/'train'
        if is_lopo:
            metrics_test = run_data.get('metrics', {})
            metrics_train = {}  # No separate train metrics in LOPO
        else:
            metrics_test = run_data.get('metrics', {}).get('test', {})
            metrics_train = run_data.get('metrics', {}).get('train', {})
        
        data_config = run_data.get('data_config', {})
        model_config = run_data.get('model_config', {})
        training_config = run_data.get('training_config', {})
        
        cm = metrics_test.get('confusion_matrix', [[0,0],[0,0]])
        
        # Calculate derived metrics
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Determine feature selection type
        feature_cols_list = data_config.get('feature_columns', [])
        feature_selection_text = "All Features"
        
        # Smart detection for detailed report
        if feature_cols_list:
             is_selected = 'LPD' in feature_cols_list and 'RPD' in feature_cols_list
             is_default = 'LPOGX' in feature_cols_list and 'RPOGX' in feature_cols_list
             is_all_features = len(feature_cols_list) > 20  # 41 features = all
             
             # Check count FIRST before checking content
             if is_all_features:
                 feature_selection_text = "All Features"
             elif is_selected:
                 feature_selection_text = "Selected (13)"
             elif is_default:
                 feature_selection_text = "All Features"
             elif len(feature_cols_list) == 13:
                 feature_selection_text = "All Features"
             else:
                 feature_selection_text = f"{len(feature_cols_list)} Features"
        
        full_config_section = self._create_full_config_section(run_data)

        # Performance cards vary based on evaluation method
        if is_lopo:
            # LOPO: Show CV results with standard deviations
            performance_cards = f"""
            <div class="card">
                <h2>LOPO Cross-Validation Results</h2>
                <p style="color: #64748b; font-size: 0.85em; margin-bottom: 12px;">
                    Leave-One-Participant-Out validation ensures generalizability to new participants
                </p>
                <div class="metric-row">
                    <span class="metric-label">Accuracy</span>
                    <span class="metric-value excellent">{metrics_test.get('accuracy', 0)*100:.2f}% +/- {metrics_test.get('accuracy_std', 0)*100:.2f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">F1-Score (Macro)</span>
                    <span class="metric-value excellent">{metrics_test.get('f1', 0):.4f} +/- {metrics_test.get('f1_std', 0):.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">ROC-AUC</span>
                    <span class="metric-value excellent">{metrics_test.get('roc_auc', 0):.4f} +/- {metrics_test.get('roc_auc_std', 0):.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Balanced Accuracy</span>
                    <span class="metric-value good">{metrics_test.get('balanced_accuracy', 0)*100:.2f}% +/- {metrics_test.get('balanced_accuracy_std', 0)*100:.2f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Class 0 F1</span>
                    <span class="metric-value">{metrics_test.get('f1_class_0', 0):.4f} +/- {metrics_test.get('f1_class_0_std', 0):.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Class 1 F1</span>
                    <span class="metric-value">{metrics_test.get('f1_class_1', 0):.4f} +/- {metrics_test.get('f1_class_1_std', 0):.4f}</span>
                </div>
            </div>
            <div class="card">
                <h2>Validation Details</h2>
                <div class="metric-row">
                    <span class="metric-label">Method</span>
                    <span class="metric-value">{run_data.get('method', 'LOPO CV')}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Total Participants</span>
                    <span class="metric-value">30 (each held out once)</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Validation Type</span>
                    <span class="metric-value">Participant-Independent</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Generalization</span>
                    <span class="metric-value" style="color: #16a34a;">[OK] High (tested on unseen participants)</span>
                </div>
                <div style="margin-top: 16px; padding: 12px; background: #f0fdf4; border-left: 3px solid #16a34a; border-radius: 4px; font-size: 0.85em;">
                    <strong style="color: #166534;">Scientific Validation:</strong><br>
                    Results represent performance on completely new participants, ensuring the model generalizes beyond the training data.
                </div>
            </div>"""
        else:
            # Simple Split: Show test and train performance
            performance_cards = f"""
            <div class="card">
                <h2>Test Performance</h2>
                <div class="metric-row">
                    <span class="metric-label">Accuracy</span>
                    <span class="metric-value excellent">{metrics_test.get('accuracy', 0)*100:.2f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Precision (PPV)</span>
                    <span class="metric-value good">{metrics_test.get('precision', 0):.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Recall (Sensitivity)</span>
                    <span class="metric-value good">{metrics_test.get('recall', 0):.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">F1-Score</span>
                    <span class="metric-value excellent">{metrics_test.get('f1', 0):.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">ROC-AUC</span>
                    <span class="metric-value excellent">{metrics_test.get('roc_auc', 0):.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Specificity</span>
                    <span class="metric-value">{specificity:.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">NPV</span>
                    <span class="metric-value">{npv:.4f}</span>
                </div>
            </div>
            
            <div class="card">
                <h2>Train Performance</h2>
                <div class="metric-row">
                    <span class="metric-label">Accuracy</span>
                    <span class="metric-value">{metrics_train.get('accuracy', 0)*100:.2f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Precision</span>
                    <span class="metric-value">{metrics_train.get('precision', 0):.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Recall</span>
                    <span class="metric-value">{metrics_train.get('recall', 0):.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">F1-Score</span>
                    <span class="metric-value">{metrics_train.get('f1', 0):.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">ROC-AUC</span>
                    <span class="metric-value">{metrics_train.get('roc_auc', 0):.4f}</span>
                </div>
            </div>"""
        
        # Sample info varies based on eval method
        if not is_lopo:
            sample_info = f"""
                <div class="metric-row">
                    <span class="metric-label">Original Trials</span>
                    <span class="metric-value">{data_config.get('original_trials', 'N/A')}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Total Samples (after cleaning)</span>
                    <span class="metric-value">{data_config.get('total_samples_used', data_config.get('n_train', 0) + data_config.get('n_test', 0))}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Train Samples</span>
                    <span class="metric-value">{data_config.get('n_train', 'N/A')}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Test Samples</span>
                    <span class="metric-value">{data_config.get('n_test', 'N/A')}</span>
                </div>"""
        else:
            sample_info = f"""
                <div class="metric-row">
                    <span class="metric-label">Participants</span>
                    <span class="metric-value">30 (each held out once)</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Cross-Validation Folds</span>
                    <span class="metric-value">30 folds (LOPO)</span>
                </div>"""
        
        # Test split only for Simple evaluation
        if not is_lopo:
            training_config_extra = f"""
                    <tr>
                        <td>Test Split</td>
                        <td>{training_config.get('test_size', 'N/A')}</td>
                    </tr>"""
        else:
            training_config_extra = ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{run_data['run_id']} - Detailed Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #f8fafc;
            color: #334155;
            padding: 16px;
            line-height: 1.3;
            font-size: 0.875rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .back-button {{
            display: inline-flex;
            align-items: center;
            padding: 8px 16px;
            background: white;
            color: #475569;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            text-decoration: none;
            margin-bottom: 16px;
            font-weight: 500;
            font-size: 0.85rem;
            transition: all 0.2s;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}
        
        .back-button:hover {{
            background: #f1f5f9;
            border-color: #cbd5e1;
            transform: translateY(-1px);
        }}
        
        .header {{
            background: white;
            padding: 16px;
            border-radius: 12px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
        }}
        
        .header h1 {{
            font-size: 1.5rem;
            margin-bottom: 4px;
            color: #0f172a;
            font-weight: 700;
            letter-spacing: -0.025em;
        }}
        
        .header p {{
            color: #64748b;
            font-size: 0.875rem;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            background: #f1f5f9;
            color: #475569;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: 600;
            margin-right: 8px;
            margin-top: 8px;
            border: 1px solid #e2e8f0;
        }}
        
        .badge-highlight {{
            background: #eff6ff;
            color: #2563eb;
            border-color: #dbeafe;
        }}
        
        .performance-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-bottom: 16px;
        }}
        
        .performance-grid .card {{
            min-width: 0;
            padding: 10px;
        }}
        
        @media (max-width: 1100px) {{
            .performance-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        
        @media (max-width: 768px) {{
            .performance-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }}
        
        .card {{
            background: white;
            padding: 12px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
        }}
        
        .card h2 {{
            font-size: 0.88rem;
            margin-bottom: 8px;
            color: #0f172a;
            font-weight: 600;
            padding-bottom: 6px;
            border-bottom: 2px solid #f1f5f9;
        }}
        
        .metric-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 0;
            border-bottom: 1px solid #f1f5f9;
        }}
        
        .metric-row:last-child {{
            border-bottom: none;
        }}
        
        .metric-label {{
            color: #64748b;
            font-weight: 500;
            font-size: 0.75rem;
        }}
        
        .metric-value {{
            font-weight: 600;
            color: #0f172a;
            font-size: 0.8rem;
            font-feature-settings: "tnum";
        }}
        
        .metric-value.excellent {{ color: #166534; }}
        .metric-value.good {{ color: #0369a1; }}
        .metric-value.fair {{ color: #854d0e; }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }}
        
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            body {{
                padding: 20px 16px;
            }}
            .header h1 {{
                font-size: 1.5rem;
            }}
        }}
        
        .chart-container {{
            background: white;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        
        .chart-container h2 {{
            font-size: 0.9rem;
            margin-bottom: 12px;
            color: #0f172a;
            font-weight: 600;
        }}
        
        .chart-container img {{
            width: 100%;
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s;
            display: block;
            margin: 0 auto;
        }}
        
        .chart-container img:hover {{
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        
        /* Modal for enlarged images */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            overflow: auto;
        }}
        
        .modal-content {{
            margin: auto;
            display: block;
            max-width: 95%;
            max-height: 95%;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }}
        
        .close {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            z-index: 1001;
        }}
        
        .close:hover,
        .close:focus {{
            color: #bbb;
        }}
        
        .config-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        
        .config-table td {{
            padding: 8px;
            border-bottom: 1px solid #f1f5f9;
        }}
        
        .config-table td:first-child {{
            font-weight: 500;
            color: #64748b;
            width: 40%;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        
        .status-badge.success {{
            background: #dcfce7;
            color: #166534;
        }}
        
        .status-badge.warning {{
            background: #fff7ed;
            color: #9a3412;
        }}
        
        @media print {{
            .back-button {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="index.html" class="back-button">? Back to Dashboard</a>
        
        <div class="header">
            <h1>{run_data['run_id']}</h1>
            <p>Detailed Performance Analysis</p>
            <div>
                <span class="badge badge-highlight">Model: {run_data['model_type'].upper()}</span>
                <span class="badge">Eval: {eval_method.upper()}</span>
                <span class="badge">Generated: {run_data.get('timestamp', 'N/A')}</span>
            </div>
        </div>
        
        <div class="performance-grid">
        {performance_cards}
        
            <div class="card">
                <h2>Data Configuration</h2>
                <div class="metric-row">
                    <span class="metric-label">Evaluation Method</span>
                    <span class="metric-value">{eval_method.upper()}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Feature Selection</span>
                    <span class="metric-value">{feature_selection_text}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Parts Filter</span>
                    <span class="metric-value">{', '.join(data_config.get('parts_filter', [])) if data_config.get('parts_filter') else 'All'}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">AOI Filter</span>
                    <span class="metric-value">{', '.join(data_config.get('aoi_filter', [])) if data_config.get('aoi_filter') else 'All'}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Time Window</span>
                    <span class="metric-value">{('Full trial' if data_config.get('time_window_s') == 'full' else f"{data_config.get('time_window_s', 'N/A')}s")}</span>
                </div>
                {sample_info}
                <div class="metric-row">
                    <span class="metric-label">Features</span>
                    <span class="metric-value">
                        {feature_selection_text} ({data_config.get('n_features', 'N/A')} raw)
                        <span style="font-size: 0.8em; color: #718096; display: block; font-weight: normal;">
                            (Expands to {model_config.get('input_dim', 'N/A')} inputs via 5 stats/feature)
                        </span>
                    </span>
                </div>
            </div>
            
            <div class="card">
                <h2>Training Configuration</h2>
                <table class="config-table">
                    <tr>
                        <td>Epochs</td>
                        <td>{training_config.get('n_epochs', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Batch Size</td>
                        <td>{training_config.get('batch_size', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Learning Rate</td>
                        <td>{training_config.get('learning_rate', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Weight Decay</td>
                        <td>{training_config.get('weight_decay', 'N/A')}</td>
                    </tr>
                    {training_config_extra}
                </table>
            </div>
        </div>
        
        {full_config_section}
        
        <div class="charts-grid">
            <div class="chart-container">
                <h2>Training History</h2>
                <img src="data:image/png;base64,{training_history_img}" alt="Training History">
            </div>
            
            <div class="chart-container">
                <h2>Confusion Matrix</h2>
                <img src="data:image/png;base64,{confusion_matrix_img}" alt="Confusion Matrix">
                <div style="margin-top: 8px; padding: 8px; background: #f7fafc; border-radius: 6px; font-size: 0.8em;">
                    TN:{cm[0][0]} FP:{cm[0][1]} | FN:{cm[1][0]} TP:{cm[1][1]}
                </div>
            </div>
            
            <div class="chart-container">
                <h2>ROC Curve</h2>
                <img src="data:image/png;base64,{roc_curve_img}" alt="ROC Curve">
            </div>
            
            <div class="chart-container">
                <h2>Class Performance</h2>
                <img src="data:image/png;base64,{class_performance_img}" alt="Class Performance">
            </div>
            
            <div class="chart-container">
                <h2>Train vs Test</h2>
                <img src="data:image/png;base64,{metrics_comparison_img}" alt="Metrics Comparison">
            </div>
            
            <div class="chart-container">
                <h2>Predictions Distribution</h2>
                <img src="data:image/png;base64,{predictions_dist_img}" alt="Predictions Distribution">
            </div>
        </div>
    </div>
    
    <!-- Modal for enlarged images -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>
    
    <script>
        function openModal(imgSrc) {{
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImage').src = imgSrc;
        }}
        
        function closeModal() {{
            document.getElementById('imageModal').style.display = 'none';
        }}
        
        // Add click event to all chart images
        document.addEventListener('DOMContentLoaded', function() {{
            const chartImages = document.querySelectorAll('.chart-container img');
            chartImages.forEach(img => {{
                img.addEventListener('click', function() {{
                    openModal(this.src);
                }});
            }});
            
            // Close modal on Escape key
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'Escape') {{
                    closeModal();
                }}
            }});
        }});
    </script>
</body>
</html>"""
    
    def _create_full_config_section(self, run_data: Dict) -> str:
        """Create a detailed HTML section with all configuration parameters."""
        config_sections = {
            'Model Configuration': run_data.get('model_config', {}),
            'Training Configuration': run_data.get('training_config', {}),
            'Data Configuration': run_data.get('data_config', {}),
        }
        
        html = '<div class="card" style="grid-column: 1 / -1; margin-bottom: 24px;">'
        html += '<h2>Full Run Configuration</h2>'
        html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 24px;">'
        
        for section_name, config_data in config_sections.items():
            if not config_data:
                continue
                
            html += f'<div><h3 style="font-size: 1rem; color: #1e293b; margin-bottom: 12px; border-bottom: 2px solid #f1f5f9; padding-bottom: 6px;">{section_name}</h3>'
            html += '<table class="config-table" style="width: 100%;">'
            
            # Sort keys for consistent display
            for key in sorted(config_data.keys()):
                value = config_data[key]
                
                # Handle lists and dicts nicely
                if isinstance(value, list):
                    if not value:
                        val_str = '<span style="color: #94a3b8;">[]</span>'
                    elif len(value) > 0 and isinstance(value[0], str):
                        # For long lists of strings (like feature names), wrap them
                        if len(value) > 5:
                            val_str = f'<div style="max-height: 100px; overflow-y: auto; font-size: 0.8em; border: 1px solid #e2e8f0; padding: 4px; border-radius: 4px;">{", ".join(value)}</div>'
                        else:
                            val_str = ', '.join(value)
                    else:
                        val_str = str(value)
                elif isinstance(value, dict):
                     # Simple recursive display for small dicts
                     val_str = '<div style="background: #f8fafc; padding: 4px; border-radius: 4px; border: 1px solid #f1f5f9;">'
                     for sub_k, sub_v in value.items():
                         val_str += f'<div style="margin-bottom: 2px;"><span style="color: #64748b; font-weight: 500;">{sub_k}:</span> {sub_v}</div>'
                     val_str += '</div>'
                else:
                    val_str = str(value)
                    
                html += f'<tr><td style="width: 40%; color: #475569; padding: 6px 0; border-bottom: 1px solid #f1f5f9;">{key}</td><td style="color: #334155; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.85em; padding: 6px 0; border-bottom: 1px solid #f1f5f9;">{val_str}</td></tr>'
            
            html += '</table></div>'
            
        html += '</div></div>'
        return html

    def _create_confusion_matrix(self, run_data: Dict) -> str:
        """Create confusion matrix visualization."""
        cm = run_data.get('metrics', {}).get('test', {}).get('confusion_matrix', [[0,0],[0,0]])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'],
                    cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        
        return self._fig_to_base64(fig)
    
    def _create_metrics_comparison(self, run_data: Dict) -> str:
        """Create train vs test metrics comparison."""
        metrics_train = run_data.get('metrics', {}).get('train', {})
        metrics_test = run_data.get('metrics', {}).get('test', {})
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        train_values = [metrics_train.get(m, 0) for m in metrics]
        test_values = [metrics_test.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, train_values, width, label='Train', color='#667eea')
        ax.bar(x + width/2, test_values, width, label='Test', color='#764ba2')
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Train vs Test Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def _create_predictions_distribution(self, run_data: Dict) -> str:
        """Create predictions distribution histogram."""
        predictions = run_data.get('metrics', {}).get('test', {}).get('predictions', [])
        
        if not predictions:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No prediction data available', 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return self._fig_to_base64(fig)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(predictions, bins=30, color='#667eea', alpha=0.7, edgecolor='black')
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        ax.set_xlabel('Predicted Probability (Class 1)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def _create_training_history(self, run_data: Dict) -> str:
        """Create training history (loss and accuracy over epochs) visualization."""
        metrics_train = run_data.get('metrics', {}).get('train', {})
        
        # Check if history exists
        history = metrics_train.get('history', {})
        if not history or 'loss' not in history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Training history not available', 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return self._fig_to_base64(fig)
        
        train_loss = history.get('loss', [])
        train_acc = history.get('accuracy', [])
        epochs = list(range(1, len(train_loss) + 1))
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot loss on left y-axis
        color_loss = '#667eea'
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold', color=color_loss)
        line1 = ax1.plot(epochs, train_loss, marker='o', linewidth=2, 
                         color=color_loss, label='Training Loss', markersize=4)
        ax1.tick_params(axis='y', labelcolor=color_loss)
        ax1.grid(True, alpha=0.3)
        
        # Create second y-axis for accuracy
        ax2 = ax1.twinx()
        
        if train_acc and len(train_acc) > 0:
            color_acc = '#48bb78'
            ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold', color=color_acc)
            line2 = ax2.plot(epochs, train_acc, marker='s', linewidth=2, 
                            color=color_acc, label='Training Accuracy', markersize=4)
            ax2.tick_params(axis='y', labelcolor=color_acc)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')
        else:
            ax1.legend(loc='best')
        
        ax1.set_title('Training History: Loss and Accuracy', fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _create_roc_curve(self, run_data: Dict) -> str:
        """Create ROC curve visualization."""
        from sklearn.metrics import roc_curve, auc
        
        predictions = run_data.get('metrics', {}).get('test', {}).get('predictions', [])
        true_labels = run_data.get('metrics', {}).get('test', {}).get('true_labels', [])
        
        if not predictions or not true_labels:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'ROC data not available', 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return self._fig_to_base64(fig)
        
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, color='#667eea', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def _create_class_performance(self, run_data: Dict) -> str:
        """Create class-wise performance breakdown."""
        cm = run_data.get('metrics', {}).get('test', {}).get('confusion_matrix', [[0,0],[0,0]])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        
        # Calculate per-class metrics
        class0_precision = tn / (tn + fn) if (tn + fn) > 0 else 0  # NPV
        class0_recall = tn / (tn + fp) if (tn + fp) > 0 else 0      # Specificity
        class0_f1 = 2 * (class0_precision * class0_recall) / (class0_precision + class0_recall) if (class0_precision + class0_recall) > 0 else 0
        
        class1_precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # PPV
        class1_recall = tp / (tp + fn) if (tp + fn) > 0 else 0      # Sensitivity
        class1_f1 = 2 * (class1_precision * class1_recall) / (class1_precision + class1_recall) if (class1_precision + class1_recall) > 0 else 0
        
        metrics = ['Precision', 'Recall', 'F1-Score']
        class0_values = [class0_precision, class0_recall, class0_f1]
        class1_values = [class1_precision, class1_recall, class1_f1]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, class0_values, width, label='Class 0', color='#48bb78')
        ax.bar(x + width/2, class1_values, width, label='Class 1', color='#667eea')
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Class-wise Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for i, (v0, v1) in enumerate(zip(class0_values, class1_values)):
            ax.text(i - width/2, v0 + 0.02, f'{v0:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, v1 + 0.02, f'{v1:.3f}', ha='center', va='bottom', fontsize=9)
        
        return self._fig_to_base64(fig)
