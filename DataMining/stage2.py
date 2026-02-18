"""
Stage 2: AOI Assignment and ta detection (Stable Fixation)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def detect_stable_fixation(group, target_aoi='Answer_Area', window_ms=1000, sampling_rate=150, coverage_threshold=0.85):
    """
    Detect first stable fixation in target AOI using sliding window.
    
    Args:
        group: Trial data (single participant + question)
        target_aoi: AOI to detect fixation in
        window_ms: Duration of stability window (milliseconds)
        sampling_rate: Eye-tracker sampling rate (Hz)
        coverage_threshold: Minimum % of samples that must remain in target AOI
    
    Returns:
        ta (relative timestamp in seconds) or NaN if not found
    """
    # Calculate window size in samples
    window_samples = int(window_ms * sampling_rate / 1000)  # 1000ms * 150Hz / 1000 = 150 samples
    
    # Sort by timestamp
    group = group.sort_values('FPOGS').reset_index(drop=True)
    trial_start = group['FPOGS'].min()
    
    # Scan samples sequentially - ONLY check windows starting in target AOI
    # Per formal definition: "For each sample where gaze falls within target AOI"
    for i in range(len(group) - window_samples + 1):
        # FIRST: Check if this sample is in target AOI
        if group.iloc[i]['AOI'] != target_aoi:
            continue  # Skip - window doesn't start in Answer_Area
        
        # Extract window (next 1000ms / 150 samples)
        window = group.iloc[i:i+window_samples]
        
        # Check stability condition: At least 85% of samples in target AOI
        same_aoi = (window['AOI'] == target_aoi).sum()
        same_aoi_pct = same_aoi / len(window)
        
        # Check if condition met (validity NOT checked - removed as per user request)
        if same_aoi_pct >= coverage_threshold:
            # Found! Return ta (relative to trial start, in seconds)
            ta = (window.iloc[0]['FPOGS'] - trial_start) / 1000.0
            return ta
    
    # Not found
    return np.nan


def generate_missing_ta_reports(ta_df, stage2_dir, config, script_dir=None):
    """
    Generate comprehensive summary reports for trials without ta detection.
    Creates both detailed CSV and summary statistics.
    
    Args:
        ta_df: DataFrame with ta results
        stage2_dir: Directory to save reports
        config: ConfigParser instance with paths and AOI coordinates
        script_dir: Script directory for resolving relative paths (optional)
    """
    no_ta_trials = ta_df[ta_df['ta'].isna()]
    
    # Get output_dir from config (not hardcoded)
    if script_dir is None:
        script_dir = Path(__file__).parent
    
    outputs_path = config.get('Paths', 'output_dir', fallback='../eye_tracking_app/outputs')
    output_dir = Path(script_dir) / outputs_path
    
    # Collect detailed statistics for each trial
    detailed_results = []
    
    for _, row in no_ta_trials.iterrows():
        participant = int(row['participant_id'].split('_')[1])
        question = int(row['question_id'].split('_')[1])
        
        csv_file = output_dir / f'participant_{participant}' / f'Q{question}.csv'
        if not csv_file.exists():
            continue
            
        df = pd.read_csv(csv_file)
        total_samples = len(df)
        
        # Read AOI coordinates from config (not hardcoded)
        aa_x1 = config.getfloat('AOI_COORDINATES', 'answer_area_x_min', fallback=0.04)
        aa_x2 = config.getfloat('AOI_COORDINATES', 'answer_area_x_max', fallback=0.76)
        aa_y1 = config.getfloat('AOI_COORDINATES', 'answer_area_y_min', fallback=0.475)
        aa_y2 = config.getfloat('AOI_COORDINATES', 'answer_area_y_max', fallback=0.935)
        
        q_x1 = config.getfloat('AOI_COORDINATES', 'question_x_min', fallback=0.05)
        q_x2 = config.getfloat('AOI_COORDINATES', 'question_x_max', fallback=0.75)
        q_y1 = config.getfloat('AOI_COORDINATES', 'question_y_min', fallback=0.025)
        q_y2 = config.getfloat('AOI_COORDINATES', 'question_y_max', fallback=0.475)
        
        # Answer_Area classification
        in_answer = ((df['BPOGY'] >= aa_y1) & (df['BPOGY'] <= aa_y2) &
                    (df['BPOGX'] >= aa_x1) & (df['BPOGX'] <= aa_x2))
        answer_samples = in_answer.sum()
        answer_pct = answer_samples / total_samples * 100
        
        # Question area
        in_question = ((df['BPOGY'] >= q_y1) & (df['BPOGY'] <= q_y2) &
                       (df['BPOGX'] >= q_x1) & (df['BPOGX'] <= q_x2))
        question_pct = in_question.sum() / total_samples * 100
        
        # Off-screen
        off_screen = ((df['BPOGX'] < 0) | (df['BPOGX'] > 1) | 
                      (df['BPOGY'] < 0) | (df['BPOGY'] > 1))
        off_screen_pct = off_screen.sum() / total_samples * 100
        
        # Longest sequence in Answer_Area
        longest_sequence = 0
        current_sequence = 0
        for val in in_answer:
            if val:
                current_sequence += 1
                longest_sequence = max(longest_sequence, current_sequence)
            else:
                current_sequence = 0
        
        # Best window coverage
        best_coverage = 0
        for i in range(len(df) - 150 + 1):
            window_coverage = in_answer.iloc[i:i+150].sum() / 150
            best_coverage = max(best_coverage, window_coverage)
        
        # Categorize failure reason
        if off_screen_pct > 50:
            failure_category = 'Off-Screen (>50%)'
        elif question_pct > 40 and answer_pct < 15:
            failure_category = 'Stayed in Question'
        elif answer_pct < 5:
            failure_category = 'Minimal Answer_Area (<5%)'
        elif longest_sequence < 75:
            failure_category = 'Short Sequences (<75 samples)'
        else:
            failure_category = 'Rapid Switching'
        
        detailed_results.append({
            'Participant': participant,
            'Question': question,
            'Total_Samples': total_samples,
            'Answer_Area_%': round(answer_pct, 1),
            'Question_%': round(question_pct, 1),
            'Off_Screen_%': round(off_screen_pct, 1),
            'Longest_Sequence': longest_sequence,
            'Best_Window_%': round(best_coverage * 100, 1),
            'Failure_Category': failure_category
        })
    
    results_df = pd.DataFrame(detailed_results)
    
    # Create summary statistics
    category_counts = results_df['Failure_Category'].value_counts()
    participant_counts = results_df.groupby('Participant').size().sort_values(ascending=False)
    
    summary_table = {
        'Metric': [
            'Total Trials',
            'Trials WITH ta',
            'Trials WITHOUT ta',
            'Detection Success Rate',
            '',
            'Failure: Off-Screen (>50%)',
            'Failure: Stayed in Question',
            'Failure: Minimal Answer_Area (<5%)',
            'Failure: Short Sequences',
            'Failure: Rapid Switching',
            '',
            'Mean Answer_Area Coverage',
            'Median Answer_Area Coverage',
            'Mean Best Window Coverage',
            'Trials Close to Success (>70%)',
            '',
            'Participant with Most Missing',
            'Second Most Missing',
            'Third Most Missing'
        ],
        'Value': [
            f"{len(ta_df)}",
            f"{len(ta_df) - len(no_ta_trials)} ({(len(ta_df) - len(no_ta_trials))/len(ta_df)*100:.1f}%)",
            f"{len(no_ta_trials)} ({len(no_ta_trials)/len(ta_df)*100:.1f}%)",
            f"{(len(ta_df) - len(no_ta_trials))/len(ta_df)*100:.1f}%",
            '',
            f"{category_counts.get('Off-Screen (>50%)', 0)} ({category_counts.get('Off-Screen (>50%)', 0)/len(no_ta_trials)*100:.1f}%)",
            f"{category_counts.get('Stayed in Question', 0)} ({category_counts.get('Stayed in Question', 0)/len(no_ta_trials)*100:.1f}%)",
            f"{category_counts.get('Minimal Answer_Area (<5%)', 0)} ({category_counts.get('Minimal Answer_Area (<5%)', 0)/len(no_ta_trials)*100:.1f}%)",
            f"{category_counts.get('Short Sequences (<75 samples)', 0)} ({category_counts.get('Short Sequences (<75 samples)', 0)/len(no_ta_trials)*100:.1f}%)",
            f"{category_counts.get('Rapid Switching', 0)} ({category_counts.get('Rapid Switching', 0)/len(no_ta_trials)*100:.1f}%)",
            '',
            f"{results_df['Answer_Area_%'].mean():.1f}%",
            f"{results_df['Answer_Area_%'].median():.1f}%",
            f"{results_df['Best_Window_%'].mean():.1f}%",
            f"{len(results_df[results_df['Best_Window_%'] > 70])} trials",
            '',
            f"participant_{participant_counts.index[0]} ({participant_counts.iloc[0]} trials)" if len(participant_counts) > 0 else 'N/A',
            f"participant_{participant_counts.index[1]} ({participant_counts.iloc[1]} trials)" if len(participant_counts) > 1 else 'N/A',
            f"participant_{participant_counts.index[2]} ({participant_counts.iloc[2]} trials)" if len(participant_counts) > 2 else 'N/A'
        ]
    }
    
    summary_df = pd.DataFrame(summary_table)
    
    # Save outputs
    summary_df.to_csv(os.path.join(stage2_dir, 'missing_ta_summary_report.csv'), index=False)
    results_df.to_csv(os.path.join(stage2_dir, 'missing_ta_detailed_breakdown.csv'), index=False)


def generate_aoi_visualizations(config, output_dir):
    """
    Generates AOI visualization images showing the layout of the screen
    and the defined Areas of Interest.
    """
    print(f"Generating AOI visualizations in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract coordinates from config (use fallbacks from OLD code)
    try:
        # Main split
        left_panel_ratio = float(config.get('DIMENSIONS', 'left_panel_ratio', fallback=0.80))
        
        # Question AOI
        q_x1 = float(config.get('AOI_COORDINATES', 'question_x_min', fallback=0.05))
        q_y1 = float(config.get('AOI_COORDINATES', 'question_y_min', fallback=0.025))
        q_x2 = float(config.get('AOI_COORDINATES', 'question_x_max', fallback=0.75))
        q_y2 = float(config.get('AOI_COORDINATES', 'question_y_max', fallback=0.475))
        
        # Answer Area AOI
        aa_x1 = float(config.get('AOI_COORDINATES', 'answer_area_x_min', fallback=0.0400))
        aa_y1 = float(config.get('AOI_COORDINATES', 'answer_area_y_min', fallback=0.4950))
        aa_x2 = float(config.get('AOI_COORDINATES', 'answer_area_x_max', fallback=0.7600))
        aa_y2 = float(config.get('AOI_COORDINATES', 'answer_area_y_max', fallback=0.9350))

    except Exception as e:
        print(f"Warning: Could not read all AOI coordinates: {e}")

    # Timer & Submit (Sidebar)
    # Timer: larger area at top of sidebar
    timer_x1 = left_panel_ratio + 0.02
    timer_y1 = 0.05
    timer_w = (1.0 - left_panel_ratio) - 0.04
    timer_h = 0.12
    
    # Submit: positioned so bottom edge aligns with answer_area_y_max
    submit_x1 = left_panel_ratio + 0.02
    submit_w = (1.0 - left_panel_ratio) - 0.04
    submit_h = 0.14
    submit_y1 = aa_y2 - submit_h  # Bottom edge aligns with answer area

    # ---------------------------------------------------------
    # PLOT 1: Overview (Question + Answer Area + Sidebar)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("AOI Definitions: Overview (Question & Answer Area)")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0) # Invert Y to match screen coordinates (0 is top)
    
    # Draw screen border
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none'))
    
    # Draw Question
    ax.add_patch(patches.Rectangle((q_x1, q_y1), q_x2-q_x1, q_y2-q_y1, linewidth=2, edgecolor='blue', facecolor='azure', alpha=0.5, label='Question'))
    ax.text(q_x1 + 0.02, q_y1 + 0.05, 'Question', fontsize=12, color='blue')
    
    # Draw Answer Area (The one the user explicitly requested)
    ax.add_patch(patches.Rectangle((aa_x1, aa_y1), aa_x2-aa_x1, aa_y2-aa_y1, linewidth=2, edgecolor='green', facecolor='honeydew', alpha=0.5, label='Answer Area'))
    ax.text(aa_x1 + 0.02, aa_y1 + 0.05, 'Answer Area\n(All Choices)', fontsize=12, color='green')
    
    # Draw Sidebar items
    ax.add_patch(patches.Rectangle((timer_x1, timer_y1), timer_w, timer_h, linewidth=2, edgecolor='red', facecolor='mistyrose', alpha=0.5, label='Timer'))
    ax.text(timer_x1, timer_y1 + timer_h/2, 'Timer', fontsize=10, color='red')
    
    ax.add_patch(patches.Rectangle((submit_x1, submit_y1), submit_w, submit_h, linewidth=2, edgecolor='orange', facecolor='papayawhip', alpha=0.5, label='Submit'))
    ax.text(submit_x1, submit_y1 + submit_h/2, 'Submit', fontsize=10, color='orange')
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'AOI_Definitions_Overview.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # PLOT 2: Detailed (Individual Choices if possible)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("AOI Definitions: Detailed (Individual Choices)")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none'))
    ax.add_patch(patches.Rectangle((q_x1, q_y1), q_x2-q_x1, q_y2-q_y1, linewidth=2, edgecolor='blue', facecolor='azure', alpha=0.3))
    
    # Draw Sidebar items
    ax.add_patch(patches.Rectangle((timer_x1, timer_y1), timer_w, timer_h, linewidth=2, edgecolor='red', facecolor='mistyrose', alpha=0.3))
    ax.add_patch(patches.Rectangle((submit_x1, submit_y1), submit_w, submit_h, linewidth=2, edgecolor='orange', facecolor='papayawhip', alpha=0.3))

    # Split Answer Area into 4 quadrants for A, B, C, D representation
    mw = (aa_x2 - aa_x1) / 2
    mh = (aa_y2 - aa_y1) / 2
    
    # A (Top-Left)
    ax.add_patch(patches.Rectangle((aa_x1, aa_y1), mw, mh, linewidth=1, edgecolor='green', facecolor='none', linestyle='--'))
    ax.text(aa_x1 + mw/2 - 0.05, aa_y1 + mh/2, 'A', fontsize=14, color='green')
    
    # B (Top-Right)
    ax.add_patch(patches.Rectangle((aa_x1 + mw, aa_y1), mw, mh, linewidth=1, edgecolor='green', facecolor='none', linestyle='--'))
    ax.text(aa_x1 + mw + mw/2 - 0.05, aa_y1 + mh/2, 'B', fontsize=14, color='green')

    # C (Bottom-Left)
    ax.add_patch(patches.Rectangle((aa_x1, aa_y1 + mh), mw, mh, linewidth=1, edgecolor='green', facecolor='none', linestyle='--'))
    ax.text(aa_x1 + mw/2 - 0.05, aa_y1 + mh + mh/2, 'C', fontsize=14, color='green')

    # D (Bottom-Right)
    ax.add_patch(patches.Rectangle((aa_x1 + mw, aa_y1 + mh), mw, mh, linewidth=1, edgecolor='green', facecolor='none', linestyle='--'))
    ax.text(aa_x1 + mw + mw/2 - 0.05, aa_y1 + mh + mh/2, 'D', fontsize=14, color='green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'AOI_Definitions_Detailed.png'))
    plt.close()

    print("AOI visualizations generated.")


def export_per_trial_aoi_data(df, output_dir):
    """
    Exports a separate CSV file for each trial (participant_id, question_id) 
    containing all data points including the assigned AOI.
    """
    print(f"\n[EXPORTING PER-TRIAL AOI DATA]")
    trials_dir = os.path.join(output_dir, 'trials_aoi_details')
    os.makedirs(trials_dir, exist_ok=True)
    
    count = 0
    grouped = df.groupby(['participant_id', 'question_id'])
    for (pid, qid), group in grouped:
        # Clean IDs for filename (participant_10 -> 10, question_1 -> 1)
        p_num = str(pid).split('_')[-1] if 'participant_' in str(pid) else str(pid)
        q_num = str(qid).split('_')[-1] if 'question_' in str(qid) else str(qid)
        
        # Define filename: e.g., p10_q2_aoi.csv
        filename = f"p{p_num}_q{q_num}_aoi.csv"
        filepath = os.path.join(trials_dir, filename)
        
        # Save to CSV
        group.to_csv(filepath, index=False)
        count += 1
        
    print(f"  Exported {count} per-trial CSV files to: {trials_dir}")


def run_stage2(df, config, report_dir):
    """
    Stage 2: AOI Assignment + Time-to-Answer (ta) detection
    
    1. Assigns AOI labels (Question, Answer_Area, Timer, Submit, Other)
    2. Detects stable fixation in Answer_Area → ta
    3. Generates visualizations
    """
    print("\n" + "="*80)
    print("STAGE 2: AOI ASSIGNMENT & TIME-TO-ANSWER (ta) DETECTION")
    print("="*80)
    
    # Read AOI coordinates from config
    left_panel_ratio = config.getfloat('DIMENSIONS', 'left_panel_ratio', fallback=0.80)
    
    # Question bounds
    q_x1 = config.getfloat('AOI_COORDINATES', 'question_x_min', fallback=0.05)
    q_x2 = config.getfloat('AOI_COORDINATES', 'question_x_max', fallback=0.75)
    q_y1 = config.getfloat('AOI_COORDINATES', 'question_y_min', fallback=0.05)
    q_y2 = config.getfloat('AOI_COORDINATES', 'question_y_max', fallback=0.50)
    
    # Answer Area bounds (entire lower half, NO gaps)
    aa_x1 = config.getfloat('AOI_COORDINATES', 'answer_area_x_min', fallback=0.03)
    aa_x2 = config.getfloat('AOI_COORDINATES', 'answer_area_x_max', fallback=0.77)
    aa_y1 = config.getfloat('AOI_COORDINATES', 'answer_area_y_min', fallback=0.50)
    aa_y2 = config.getfloat('AOI_COORDINATES', 'answer_area_y_max', fallback=0.95)
    
    # Timer bounds (sidebar, top portion)
    timer_x1 = left_panel_ratio + 0.02
    timer_x2 = 1.0 - 0.02
    timer_y1 = 0.05
    timer_y2 = 0.17  # 0.12 height
    
    # Submit bounds (sidebar, aligned with answer area bottom)
    submit_x1 = left_panel_ratio + 0.02
    submit_x2 = 1.0 - 0.02
    submit_y2 = aa_y2
    submit_y1 = submit_y2 - 0.14  # 0.14 height
    
    print(f"\n[AOI COORDINATES]")
    print(f"  Question: x=[{q_x1:.4f}, {q_x2:.4f}], y=[{q_y1:.4f}, {q_y2:.4f}]")
    print(f"  Answer Area: x=[{aa_x1:.4f}, {aa_x2:.4f}], y=[{aa_y1:.4f}, {aa_y2:.4f}] (entire lower half)")
    print(f"  Timer: x=[{timer_x1:.4f}, {timer_x2:.4f}], y=[{timer_y1:.4f}, {timer_y2:.4f}]")
    print(f"  Submit: x=[{submit_x1:.4f}, {submit_x2:.4f}], y=[{submit_y1:.4f}, {submit_y2:.4f}]")
    
    # Assign AOI labels
    print(f"\n[ASSIGNING AOI LABELS]")
    def assign_aoi(row):
        x, y = row['BPOGX'], row['BPOGY']
        
        # Check Question
        if q_x1 <= x <= q_x2 and q_y1 <= y <= q_y2:
            return 'Question'
        
        # Check Answer Area (entire lower half, no gaps)
        if aa_x1 <= x <= aa_x2 and aa_y1 <= y <= aa_y2:
            return 'Answer_Area'
        
        # Check Timer
        if timer_x1 <= x <= timer_x2 and timer_y1 <= y <= timer_y2:
            return 'Timer'
        
        # Check Submit
        if submit_x1 <= x <= submit_x2 and submit_y1 <= y <= submit_y2:
            return 'Submit'
        
        return 'Other'
    
    df['AOI'] = df.apply(assign_aoi, axis=1)
    
    # AOI distribution
    aoi_counts = df['AOI'].value_counts()
    total_samples = len(df)
    print(f"  Total samples: {total_samples:,}")
    for aoi, count in aoi_counts.items():
        pct = count / total_samples * 100
        print(f"    {aoi}: {count:,} ({pct:.1f}%)")
    
    # Detect ta (stable fixation in Answer_Area)
    print(f"\n[DETECTING STABLE FIXATION - ta]")
    window_ms = config.getint('STAGE2_TA', 'ta_window_ms', fallback=1000)
    sampling_rate = config.getint('Analysis', 'gaze_sampling_rate_hz', fallback=150)
    coverage_threshold = config.getfloat('STAGE2_TA', 'ta_answer_coverage_threshold', fallback=0.85)
    
    print(f"  Window: {window_ms}ms (~{int(window_ms * sampling_rate / 1000)} samples)")
    print(f"  Coverage threshold: {coverage_threshold * 100:.0f}%")
    
    ta_results = []
    for (participant_id, question_id), group in df.groupby(['participant_id', 'question_id']):
        ta = detect_stable_fixation(
            group, 
            target_aoi='Answer_Area',
            window_ms=window_ms,
            sampling_rate=sampling_rate,
            coverage_threshold=coverage_threshold
        )
        
        # Get part from the group (should be same for all samples in trial)
        part = group['part'].iloc[0] if 'part' in group.columns else None
        
        ta_results.append({
            'participant_id': participant_id,
            'question_id': question_id,
            'ta': ta,
            'part': part
        })
    
    ta_df = pd.DataFrame(ta_results)
    
    # Statistics
    valid_ta = ta_df['ta'].notna().sum()
    missing_ta = ta_df['ta'].isna().sum()
    print(f"  Trials with ta: {valid_ta}/{len(ta_df)} ({valid_ta/len(ta_df)*100:.1f}%)")
    print(f"  Trials without ta: {missing_ta}")
    
    if valid_ta > 0:
        print(f"  ta statistics (seconds):")
        print(f"    Mean: {ta_df['ta'].mean():.2f}s")
        print(f"    Median: {ta_df['ta'].median():.2f}s")
        print(f"    Min: {ta_df['ta'].min():.2f}s")
        print(f"    Max: {ta_df['ta'].max():.2f}s")
    
    # Generate visualizations
    print(f"\n[GENERATING VISUALIZATIONS]")
    stage2_dir = os.path.join(report_dir, 'stage2')
    generate_aoi_visualizations(config, stage2_dir)
    
    # Save ta results
    ta_output_path = os.path.join(stage2_dir, 'ta_per_trial.csv')
    
    try:
        ta_df.to_csv(ta_output_path, index=False)
        print(f"  Saved: {ta_output_path}")
    except PermissionError:
        print(f"  [WARNING] Could not save {ta_output_path} because it is open in another program.")
        print(f"  Proceeding with in-memory data (Stage 3 will still work).")

    # Export per-trial AOI details
    export_per_trial_aoi_data(df, stage2_dir)
    
    # Generate comprehensive reports for trials without ta
    if missing_ta > 0:
        print(f"\n[GENERATING MISSING TA REPORTS]")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        generate_missing_ta_reports(ta_df, stage2_dir, config, script_dir)
        print(f"  Reports saved in: {stage2_dir}")
    
    # Collect statistics for reporting
    stats = {
        'total_trials': len(ta_df),
        'trials_with_ta': valid_ta,
        'trials_without_ta': missing_ta,
        'ta_detection_rate': valid_ta / len(ta_df) * 100 if len(ta_df) > 0 else 0,
        'ta_mean': ta_df['ta'].mean() if valid_ta > 0 else 0,
        'ta_median': ta_df['ta'].median() if valid_ta > 0 else 0
    }
    
    print("\n[STAGE 2 COMPLETE]")
    return df, ta_df, stats
