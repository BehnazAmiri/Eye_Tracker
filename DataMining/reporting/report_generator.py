"""
HTML Report Generator for Data Mining Pipeline
Generates comprehensive reports with comparison tables for different threshold configurations
"""

from datetime import datetime
from pathlib import Path


class ReportGenerator:
    """Generates HTML reports for data mining pipeline results."""
    
    def __init__(self, output_dir='results/reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, config, stage1_stats, stage2_stats, stage3_stats, comparison_data=None):
        """
        Generate comprehensive HTML report.
        
        Args:
            config: ConfigParser object
            stage1_stats: Stage 1 statistics dictionary
            stage2_stats: Stage 2 statistics dictionary
            stage3_stats: Stage 3 statistics dictionary
            comparison_data: Optional comparison data from ThresholdComparison
        """
        html_content = self._generate_html(config, stage1_stats, stage2_stats, stage3_stats, comparison_data)
        
        # Save report
        report_path = self.output_dir / 'pipeline_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n  [OK] Report saved: {report_path}")
        return report_path
    
    def _generate_html(self, config, stage1_stats, stage2_stats, stage3_stats, comparison_data):
        """Generate HTML content."""
        
        # Extract threshold values
        stage1_invalid_threshold = config.getfloat('Analysis', 'stage1_invalid_pct_threshold', fallback=0.30)
        stage1_participant_threshold = config.getfloat('Analysis', 'stage1_participant_exclusion_threshold', fallback=0.50)
        ta_window_ms = config.getint('STAGE2_TA', 'ta_window_ms', fallback=1000)
        ta_coverage = config.getfloat('STAGE2_TA', 'ta_answer_coverage_threshold', fallback=0.85)
        stage3_percentile = config.getint('Analysis', 'stage3_threshold_percentile', fallback=25)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Mining Pipeline Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .stage {{
            background: white;
            padding: 25px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stage h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .stage h3 {{
            color: #764ba2;
            margin-top: 25px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .metric-box {{
            display: inline-block;
            background: #f8f9fa;
            padding: 15px 20px;
            margin: 10px 10px 10px 0;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        .threshold-info {{
            background: #e7f3ff;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #2196F3;
        }}
        .description {{
            color: #555;
            line-height: 1.6;
            margin: 15px 0;
        }}
        ul {{
            color: #555;
            line-height: 1.8;
        }}
        .comparison-section {{
            background: #fff9e6;
            padding: 20px;
            border-radius: 10px;
            margin: 30px 0;
            border: 2px solid #ffd700;
        }}
        .comparison-section h2 {{
            color: #f57c00;
            margin-top: 0;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 50px;
            padding: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Data Mining Pipeline Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
        
        # Stage 1 Section
        html += f"""
    <div class="stage">
        <h2>Stage 1: Identification of Invalid Gaze Samples</h2>
        
        <div class="description">
            Each gaze sample is evaluated using the tracker's validity indicators:
        </div>
        
        <ul>
            <li><strong>If BPOGV = 1</strong>, the sample is considered valid and kept.</li>
            <li><strong>If BPOGV = 0</strong>, the blink indicator (BKID) is checked:
                <ul>
                    <li>If BKID = 1, valid and kept.</li>
                    <li>If BKID = 0, the sample is marked as invalid.</li>
                </ul>
            </li>
        </ul>
        
        <h3>Trial-Level Exclusion</h3>
        <div class="threshold-info">
            <strong>Threshold:</strong> invalid &gt; {stage1_invalid_threshold:.0%} (or equivalently valid &lt; {1-stage1_invalid_threshold:.0%})<br>
            <strong>Action:</strong> If a trial exceeds the invalid percentage threshold, the entire trial is excluded from processing.<br>
            <strong>Temporal Integrity:</strong> Individual samples are NOT deleted. Only entire trials are excluded.
        </div>
        
        <h3>Participant-Level Exclusion</h3>
        <div class="threshold-info">
            <strong>Threshold:</strong> excluded trials &gt; {stage1_participant_threshold:.0%}<br>
            <strong>Action:</strong> If a participant has more than {stage1_participant_threshold:.0%} of their trials excluded, 
            the entire participant (all their trials, even good ones) is excluded from processing.
        </div>
        
        <h3>Results:</h3>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Samples</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Total Trials</strong></td>
                    <td>{stage1_stats['total_trials']}</td>
                    <td>100.0%</td>
                    <td>{stage1_stats['total_samples']:,}</td>
                </tr>
                <tr>
                    <td>Trials with &gt;{stage1_invalid_threshold:.0%} corruption (trial-level exclusion)</td>
                    <td>{stage1_stats['excluded_trials_trial_level']}</td>
                    <td>{stage1_stats['excluded_trials_trial_level_pct']:.1f}%</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Participants with &gt;{stage1_participant_threshold:.0%} bad trials (fully excluded)</td>
                    <td>{stage1_stats['excluded_participants']}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td style="padding-left: 30px;">↳ Additional trials lost due to participant removal</td>
                    <td>{stage1_stats['excluded_trials_participant_level']}</td>
                    <td>{stage1_stats['excluded_trials_participant_level_pct']:.1f}%</td>
                    <td>-</td>
                </tr>
                <tr style="background-color: #e8f5e9;">
                    <td><strong>Total Excluded Trials</strong></td>
                    <td><strong>{stage1_stats['excluded_trials']}</strong></td>
                    <td><strong>{stage1_stats['excluded_trials_pct']:.1f}%</strong></td>
                    <td><strong>{stage1_stats['excluded_samples']:,}</strong></td>
                </tr>
                <tr style="background-color: #c8e6c9;">
                    <td><strong>Kept Trials (passed filters)</strong></td>
                    <td><strong>{stage1_stats['kept_trials']}</strong></td>
                    <td><strong>{stage1_stats['kept_trials_pct']:.1f}%</strong></td>
                    <td><strong>{stage1_stats['kept_samples']:,}</strong></td>
                </tr>
            </tbody>
        </table>
    </div>
"""
        
        # Stage 2 Section
        html += f"""
    <div class="stage">
        <h2>Stage 2: AOI Assignment</h2>
        
        <div class="description">
            <strong>ta (First Stable Fixation):</strong> The timestamp when the participant first fixates on any Answer AOI 
            for a stable duration.
        </div>
        
        <ul>
            <li><strong>Time Window:</strong> {ta_window_ms} milliseconds ({ta_window_ms/1000:.1f} second{'s' if ta_window_ms != 1000 else ''}) of consecutive samples</li>
            <li><strong>Sampling Rate:</strong> 150 Hz → 150 samples per second</li>
            <li><strong>Coverage Threshold:</strong> At least {ta_coverage:.0%} of samples within the {ta_window_ms/1000:.1f}-second window must remain in the target AOI</li>
        </ul>
        
        <h3>Detection Process:</h3>
        <ol>
            <li>Start from trial beginning, scan samples sequentially</li>
            <li>For each sample where gaze falls within target AOI (ta):
                <ul>
                    <li>Extract the next {ta_window_ms}ms (≈{int(ta_window_ms * 0.15)} samples) as candidate window</li>
                    <li>Check: Do ≥{ta_coverage:.0%} of samples in this window remain in the same AOI?</li>
                    <li>If YES → Mark this timestamp as the stable fixation landmark</li>
                    <li>If NO → Continue scanning to next sample</li>
                </ul>
            </li>
            <li>First successful detection becomes the temporal landmark</li>
        </ol>
        
        <h3>Results:</h3>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Total Trials</strong></td>
                    <td>{stage2_stats['total_trials']}</td>
                </tr>
                <tr style="background-color: #e8f5e9;">
                    <td>Trials with ta</td>
                    <td>{stage2_stats['trials_with_ta']} ({stage2_stats['ta_detection_rate']:.1f}%)</td>
                </tr>
                <tr style="background-color: #ffebee;">
                    <td>Trials without ta</td>
                    <td>{stage2_stats['trials_without_ta']} ({100-stage2_stats['ta_detection_rate']:.1f}%)</td>
                </tr>
                <tr>
                    <td><strong>Detection Success Rate</strong></td>
                    <td><strong>{stage2_stats['trials_with_ta']} ({stage2_stats['ta_detection_rate']:.1f}%)</strong></td>
                </tr>
            </tbody>
        </table>
    </div>
"""
        
        # Stage 3 Section
        html += f"""
    <div class="stage">
        <h2>Stage 3: Randomness Labeling</h2>
        
        <div class="description">
            This stage classifies each trial as <span class="highlight">RANDOM</span> or <span class="highlight">NOT_RANDOM</span> 
            based on time from first answer fixation (ta) to response (end of the trial).
        </div>
        
        <h3>Classification Method:</h3>
        <ul>
            <li><strong>Timer-No-Correct:</strong> ALL trials classified as RANDOM (no threshold applied). 
            These trials had timer pressure AND no correct answer available, making strategic decision-making impossible.</li>
            <li><strong>No-Timer & Timer-Correct:</strong> Uses {stage3_percentile}th percentile (P{stage3_percentile}) threshold calculated separately for each part:
                <ul>
                    <li><strong>NOT_RANDOM:</strong> Time from t_answer ≥ P{stage3_percentile} → Strategic, deliberate decision-making</li>
                    <li><strong>RANDOM:</strong> Time from t_answer &lt; P{stage3_percentile} → Fast, impulsive response</li>
                </ul>
            </li>
        </ul>
        
        <div class="threshold-info">
            <strong>t_answer</strong> is the decision time after attention reaches the answers: 
            it is computed as <strong>t_end − ta</strong>, where ta is the first stable fixation on the Answer Area 
            and t_end is the last sample.
        </div>
        
        <div class="description">
            <ul>
                <li>To avoid using an arbitrary threshold, we applied a <strong>data-driven approach</strong> based on 
                the {stage3_percentile}th percentile (P{stage3_percentile}) of response time.</li>
                <li>P{stage3_percentile} represents the fastest {stage3_percentile}% of valid responses and is used to 
                identify unusually quick, potentially impulsive answers.</li>
                <li>Trials with response time below P{stage3_percentile} are labeled as RANDOM, while longer responses 
                are labeled as NOT_RANDOM, indicating more deliberate processing.</li>
            </ul>
        </div>
        
        <h3>Results:</h3>
        <div class="metric-box">
            <div class="metric-label">Total Trials</div>
            <div class="metric-value">{stage3_stats['total_trials']}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Total Valid Trials</div>
            <div class="metric-value">{stage3_stats['total_valid']}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">NOT_RANDOM</div>
            <div class="metric-value" style="color: #4CAF50;">{stage3_stats['not_random']}<br>
            <small>({stage3_stats['not_random_pct']:.1f}% of valid | {stage3_stats['not_random']/stage3_stats['total_trials']*100:.1f}% of all)</small></div>
        </div>
        <div class="metric-box">
            <div class="metric-label">RANDOM</div>
            <div class="metric-value" style="color: #FF9800;">{stage3_stats['random']}<br>
            <small>({stage3_stats['random_pct']:.1f}% of valid | {stage3_stats['random']/stage3_stats['total_trials']*100:.1f}% of all)</small></div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Invalid</div>
            <div class="metric-value" style="color: #F44336;">{stage3_stats['invalid']}<br>
            <small>({stage3_stats['invalid_pct']:.1f}% of all)</small></div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Part</th>
                    <th>NOT_RANDOM</th>
                    <th>RANDOM</th>
                    <th>Invalid</th>
                    <th>Total</th>
                    <th>NOT_RANDOM %</th>
                    <th>RANDOM %</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add part-wise breakdown
        for part_name, part_data in stage3_stats.get('by_part', {}).items():
            valid_total = part_data['not_random'] + part_data['random']
            not_random_pct = (part_data['not_random'] / valid_total * 100) if valid_total > 0 else 0
            random_pct = (part_data['random'] / valid_total * 100) if valid_total > 0 else 0
            
            html += f"""
                <tr>
                    <td><strong>{part_name}</strong></td>
                    <td>{part_data['not_random']}</td>
                    <td>{part_data['random']}</td>
                    <td>{part_data['invalid']}</td>
                    <td>{part_data['total']}</td>
                    <td>{not_random_pct:.1f}%</td>
                    <td>{random_pct:.1f}%</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
"""
        
        # Comparison Section (if data available)
        if comparison_data:
            html += self._generate_comparison_section(comparison_data)
        
        # Footer
        html += """
    <div class="footer">
        <p>Generated by Data Mining Pipeline</p>
        <p>© 2026 Master Thesis Project</p>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_comparison_section(self, comparison_data):
        """Generate HTML for unified threshold comparison table."""
        html = """
    <div class="comparison-section">
        <h2>🔍 Threshold Comparison Across Multiple Runs</h2>
        <p>Each run represents a complete pipeline execution with specific threshold configurations. Compare how different parameters affect the results across all stages.</p>

        <table>
            <thead>
                <tr>
"""
        # Headers
        for header in comparison_data['headers']:
            html += f"                    <th>{header}</th>\n"
        html += """
                </tr>
            </thead>
            <tbody>
"""
        # Rows with different styling based on type
        for row_data in comparison_data['rows']:
            row_type = row_data['type']
            row_name = row_data['name']
            row_values = row_data['values']
            
            # Apply different styling for different row types
            if row_type == 'section':
                # Section header rows (bold, colored background)
                html += f'                <tr style="background-color: #667eea; color: white; font-weight: bold;">\n'
                html += f'                    <td colspan="{len(comparison_data["headers"])}">{row_name}</td>\n'
                html += '                </tr>\n'
            elif row_type == 'separator':
                # Empty separator rows
                html += f'                <tr style="height: 10px;">\n'
                html += f'                    <td colspan="{len(comparison_data["headers"])}" style="border: none; background: none;"></td>\n'
                html += '                </tr>\n'
            elif row_type == 'threshold':
                # Threshold rows (light yellow background)
                html += '                <tr style="background-color: #fffacd;">\n'
                html += f'                    <th style="text-align: left; padding-left: 20px;">{row_name}</th>\n'
                for value in row_values:
                    html += f'                    <td style="font-weight: 500;">{value}</td>\n'
                html += '                </tr>\n'
            else:  # data
                # Regular data rows
                html += '                <tr>\n'
                html += f'                    <th style="text-align: left; padding-left: 30px; font-weight: normal;">{row_name}</th>\n'
                for value in row_values:
                    html += f'                    <td>{value}</td>\n'
                html += '                </tr>\n'
        
        html += """
            </tbody>
        </table>
    </div>
"""
        return html
