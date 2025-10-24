print("--- Script execution started ---")
import os
import glob
import json
import pandas as pd
import numpy as np
import configparser
import tkinter as tk
from tkinter import simpledialog
import textwrap
import time
import threading
import queue
from tkinter import ttk
import matplotlib
import sys
matplotlib.use('Agg')  # Use non-GUI backend for thread-safe figure creation

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    import matplotlib.patheffects as pe
except ImportError:
    print("="*60)
    print(" ERROR: Required visualization libraries are not installed.")
    print(" Please run the following command in your terminal to install them:")
    print("\n   pip install matplotlib seaborn\n")
    print("="*60)
    exit()

# ======================================================================================
# >>> ADDED: LOGGING & UI HELPERS
# ======================================================================================
class Logger:
    def __init__(self, progress_queue, original_stdout):
        self.progress_queue = progress_queue
        self.original_stdout = original_stdout

    def write(self, text):
        self.original_stdout.write(text)
        stripped_text = text.strip()
        if self.progress_queue and stripped_text:
            self.progress_queue.put(("log", stripped_text))

    def flush(self):
        self.original_stdout.flush()

monitor = None  # Global monitor instance

# ======================================================================================
# STAGE 0: CONFIGURATION AND DATA LOADING
# ======================================================================================

def load_config(config_path='config.ini'):
    """Loads configuration from the INI file."""
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file not found at {config_path}. Using default paths.")
        # Create a default config object if file doesn't exist
        config['Paths'] = {'output_dir': 'outputs',
                           'questions_json': 'questions.json',
                           'question_exams_dir': 'question_exams'}
        # Analysis defaults
        config['Analysis'] = {'time_cap_s': '', 'exclude_censored': 'false'}
        # Overlay defaults kept empty; we fallback later
        return config
    config.read(config_path)

    # Ensure sections and defaults exist
    if 'Paths' not in config:
        config['Paths'] = {}
    if 'Analysis' not in config:
        config['Analysis'] = {}

    config['Paths'].setdefault('output_dir', 'outputs')
    config['Paths'].setdefault('questions_json', 'questions.json')
    config['Paths'].setdefault('question_exams_dir', 'question_exams')
    # Analysis defaults
    config['Analysis'].setdefault('time_cap_s', '')
    config['Analysis'].setdefault('exclude_censored', 'false')
    # New diagnostics and thresholds
    config['Analysis'].setdefault('invalid_gaze_threshold', '0.2')
    config['Analysis'].setdefault('consecutive_zero_threshold', '5')
    config['Analysis'].setdefault('lb_multiplier', '1.5')

    return config


def load_all_participant_data(output_dir):
    """Loads and concatenates all participant CSV files from the output directory."""
    all_files = glob.glob(os.path.join(output_dir, 'participant_*', 'Q*.csv'))
    if not all_files:
        print(f"Warning: No question CSV files (Q*.csv) found in {output_dir}. Please check the path in config.ini.")
        return pd.DataFrame()
        
    df_list = []
    for file in all_files:
        try:
            participant_id = os.path.basename(os.path.dirname(file))
            question_id = os.path.basename(file).replace('.csv', '')
            df = pd.read_csv(file)
            df['participant_id'] = participant_id
            df['question_id'] = question_id
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)


def load_question_data(questions_path, output_dir):
    """Loads question metadata and participant answers from answers.json files."""
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
        correct_answers = {}
        for q in questions:
            question_id_str = f"Q{q['id']}"
            # Determine correct option by checking option id suffix '-C' or last component == 'C'
            for idx, option in enumerate(q.get('options', []), start=1):
                opt_id = option.get('id', '')
                parts = opt_id.split('-')
                if parts and parts[-1] == 'C':
                    # Map numeric index to letter (1->A, 2->B, ...)
                    letter = chr(ord('A') + idx - 1) if idx <= 4 else None
                    if letter:
                        correct_answers[question_id_str] = letter
                    else:
                        # fallback: keep original id if we cannot map
                        correct_answers[question_id_str] = opt_id
                    break
    
    answer_files = glob.glob(os.path.join(output_dir, 'participant_*', 'answers.json'))
    answers_list = []
    for file in answer_files:
        try:
            participant_id = os.path.basename(os.path.dirname(file))
            with open(file, 'r', encoding='utf-8') as f:
                participant_answers = json.load(f)
                # It's a list of objects, not a dictionary
                for answer in participant_answers:
                    # The question_id in answers.json is an integer
                    question_id_key = f"Q{answer['question_id']}"
                    chosen_option = answer.get('chosen_option')

                    # Derive chosen letter from chosen_option if possible (format: Q-<idx>[-C])
                    chosen_letter = None
                    if isinstance(chosen_option, str):
                        parts = chosen_option.split('-')
                        if len(parts) >= 2:
                            try:
                                idx = int(parts[1])
                                chosen_letter = chr(ord('A') + idx - 1)
                            except Exception:
                                chosen_letter = None

                    # Determine correctness: prefer comparing letters (new mapping), otherwise fallback to id equality
                    correct_val = correct_answers.get(question_id_key)
                    is_correct = 0
                    if chosen_letter and isinstance(correct_val, str) and len(correct_val) == 1 and correct_val.isalpha():
                        is_correct = 1 if chosen_letter.upper() == correct_val.upper() else 0
                    else:
                        # Fallback: compare raw chosen_option id to stored correct id (legacy)
                        is_correct = 1 if chosen_option == correct_val else 0

                    answers_list.append({
                        'participant_id': participant_id,
                        'question_id': question_id_key,
                        'is_correct': is_correct
                    })
        except Exception as e:
            print(f"Error processing answer file {file}: {e}")
                
    return pd.DataFrame(answers_list), correct_answers


def load_question_parts(question_exams_dir, output_dir):
    """Loads which questions belong to Part 1 and Part 2 for each participant."""
    part_mapping = []
    participant_dirs = glob.glob(os.path.join(output_dir, 'participant_*'))
    
    for p_dir in participant_dirs:
        participant_id = os.path.basename(p_dir)
        p_id_numeric = participant_id.split('_')[-1]
        exam_file = os.path.join(question_exams_dir, f"Participant_{p_id_numeric}.json")
        
        if not os.path.exists(exam_file):
            print(f"Warning: Exam file not found for {participant_id} at {exam_file}")
            continue
            
        with open(exam_file, 'r', encoding='utf-8') as f:
            exam_data = json.load(f)
            part_mapping.extend([
                {'participant_id': participant_id, 'question_id': f"Q{item['question_id']}", 'part': 'Part 1'}
                for item in exam_data.get('Part1', [])
            ])
            part_mapping.extend([
                {'participant_id': participant_id, 'question_id': f"Q{item['question_id']}", 'part': 'Part 2'}
                for item in exam_data.get('Part2', [])
            ])
                
    return pd.DataFrame(part_mapping)


def question_part_distribution(question_exams_dir, output_dir):
    """Return DataFrame with counts of how many participants saw each question in Part1 vs Part2."""
    rows = []
    participant_dirs = glob.glob(os.path.join(output_dir, 'participant_*'))
    for p_dir in participant_dirs:
        participant_id = os.path.basename(p_dir)
        p_id_numeric = participant_id.split('_')[-1]
        exam_file = os.path.join(question_exams_dir, f"Participant_{p_id_numeric}.json")
        if not os.path.exists(exam_file):
            continue
        try:
            with open(exam_file, 'r', encoding='utf-8') as f:
                qdata = json.load(f)
                for item in qdata.get('Part1', []):
                    rows.append({'question_id': f"Q{item['question_id']}", 'part': 'Part 1', 'participant_id': participant_id})
                for item in qdata.get('Part2', []):
                    rows.append({'question_id': f"Q{item['question_id']}", 'part': 'Part 2', 'participant_id': participant_id})
        except Exception:
            pass
    if not rows:
        return pd.DataFrame(columns=['question_id', 'count_part1', 'count_part2', 'total'])
    df = pd.DataFrame(rows)
    part_counts = df.pivot_table(index='question_id', columns='part', values='participant_id', aggfunc='nunique', fill_value=0)
    part_counts = part_counts.rename(columns={'Part 1':'count_part1', 'Part 2':'count_part2'}) if 'Part 1' in part_counts.columns or 'Part 2' in part_counts.columns else part_counts
    part_counts['count_part1'] = part_counts.get('count_part1', 0)
    part_counts['count_part2'] = part_counts.get('count_part2', 0)
    part_counts['total'] = part_counts['count_part1'] + part_counts['count_part2']
    return part_counts.reset_index()


def load_question_texts(questions_path):
    """Loads question and answer texts from the questions.json file."""
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    texts = {}
    for q in questions_data:
        question_id = f"Q{q['id']}"
        texts[question_id] = {
            'text': q['question'],
            'options': [{'id': opt['id'], 'text': opt['text']} for opt in q['options']]
        }
    return texts


def get_input_range():
    """Creates a GUI to get participant and question ranges."""
    import configparser
    import os
    # Determine config path (same as main)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.ini')
    config = configparser.ConfigParser()
    if os.path.exists(config_path):
        config.read(config_path)
    # Load previous input values if available
    prev_participant = "1-30"
    prev_question = "1-15"
    prev_plots = {
        "summary_plots": True,
        "heatmaps": False,
        "scatterplots": False,
        "aoi_summary_per_question": True,
        "aoi_per_question": True,
        "aoi_per_label": True,
        "save_stage_outputs": True
    }
    if 'Input' in config:
        prev_participant = config['Input'].get('participant_range', prev_participant)
        prev_question = config['Input'].get('question_range', prev_question)
        for k in prev_plots:
            if k in config['Input']:
                prev_plots[k] = config['Input'].getboolean(k, fallback=prev_plots[k])

    root = tk.Tk()
    root.title("Input for Analysis")
    root.geometry("340x480")

    tk.Label(root, text="Participant Range (e.g., 1-30):").pack(pady=5)
    participant_entry = tk.Entry(root)
    participant_entry.pack(pady=5)
    participant_entry.insert(0, prev_participant)

    tk.Label(root, text="Question Range (e.g., 1-15):").pack(pady=5)
    question_entry = tk.Entry(root)
    question_entry.pack(pady=5)
    question_entry.insert(0, prev_question)

    # Checkboxes for plot selection
    plot_vars = {
        "summary_plots": tk.BooleanVar(value=prev_plots["summary_plots"]),
        "heatmaps": tk.BooleanVar(value=prev_plots["heatmaps"]),
        "scatterplots": tk.BooleanVar(value=prev_plots["scatterplots"]),
        "aoi_summary_per_question": tk.BooleanVar(value=prev_plots["aoi_summary_per_question"]),
        "aoi_per_question": tk.BooleanVar(value=prev_plots["aoi_per_question"]),
        "aoi_per_label": tk.BooleanVar(value=prev_plots["aoi_per_label"]),
        "save_stage_outputs": tk.BooleanVar(value=prev_plots["save_stage_outputs"])
    }

    tk.Label(root, text="Select Plots to Generate:").pack(pady=5)
    tk.Checkbutton(root, text="AOI Time Summary per Question", variable=plot_vars["aoi_summary_per_question"]).pack(anchor='w', padx=20, pady=2)
    tk.Checkbutton(root, text="Pipeline Summary Plots (Stage 1-4)", variable=plot_vars["summary_plots"]).pack(anchor='w', padx=20, pady=(20, 2))
    tk.Checkbutton(root, text="Gaze Heatmaps (per participant/question)", variable=plot_vars["heatmaps"]).pack(anchor='w', padx=20, pady=2)
    tk.Checkbutton(root, text="Gaze Scatterplots (per participant/question)", variable=plot_vars["scatterplots"]).pack(anchor='w', padx=20, pady=2)
    tk.Checkbutton(root, text="AOI Time per Question (Bar Chart)", variable=plot_vars["aoi_per_question"]).pack(anchor='w', padx=20, pady=2)
    tk.Checkbutton(root, text="AOI Time per Label (Bar Chart)", variable=plot_vars["aoi_per_label"]).pack(anchor='w', padx=20, pady=2)
    tk.Checkbutton(root, text="Save intermediate stage output files", variable=plot_vars["save_stage_outputs"]).pack(anchor='w', padx=20, pady=8)

    ranges = {}
    selected_plots = {}

    def on_submit():
        p_range_str = participant_entry.get()
        q_range_str = question_entry.get()
        try:
            p_start, p_end = map(int, p_range_str.split('-'))
            ranges['participants'] = range(p_start, p_end + 1)
        except (ValueError, IndexError):
            ranges['participants'] = None
        try:
            q_start, q_end = map(int, q_range_str.split('-'))
            ranges['questions'] = range(q_start, q_end + 1)
        except (ValueError, IndexError):
            ranges['questions'] = None
        for key, var in plot_vars.items():
            selected_plots[key] = var.get()
        # Save input values to config
        if 'Input' not in config:
            config['Input'] = {}
        config['Input']['participant_range'] = p_range_str
        config['Input']['question_range'] = q_range_str
        for key, val in selected_plots.items():
            config['Input'][key] = str(val)
        with open(config_path, 'w') as configfile:
            config.write(configfile)
        root.destroy()

    def on_default():
        # Reset all fields to initial default values
        participant_entry.delete(0, 'end')
        participant_entry.insert(0, "1-30")
        question_entry.delete(0, 'end')
        question_entry.insert(0, "1-15")
        plot_vars["summary_plots"].set(True)
        plot_vars["heatmaps"].set(False)
        plot_vars["scatterplots"].set(False)
        plot_vars["aoi_summary_per_question"].set(True)
        plot_vars["aoi_per_question"].set(True)
        plot_vars["aoi_per_label"].set(True)
        plot_vars["save_stage_outputs"].set(True)

    def on_restore():
        # Restore all fields to last saved values from config
        # Use prev_participant, prev_question, prev_plots from earlier
        participant_entry.delete(0, 'end')
        participant_entry.insert(0, prev_participant)
        question_entry.delete(0, 'end')
        question_entry.insert(0, prev_question)
        for k in plot_vars:
            plot_vars[k].set(prev_plots[k])

    # افزودن دکمه حذف خروجی‌ها
    def delete_outputs():
        import shutil
        import tkinter.messagebox as messagebox
        
        # Confirmation dialog
        if not messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete all output files and folders? This action cannot be undone."):
            return # User cancelled deletion

        script_dir = os.path.dirname(os.path.abspath(__file__))
        targets = [
            os.path.join(script_dir, 'processed_data'),
            os.path.join(script_dir, 'intermediate_processed_data'),
            os.path.join(script_dir, 'visualizations'),
            os.path.join(script_dir, 'reports'),
        ]
        errors = []
        for path in targets:
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    errors.append(f"{path}: {e}")
        if errors:
            messagebox.showerror("Delete Outputs", "Some items could not be deleted:\n" + '\n'.join(errors))
        else:
            messagebox.showinfo("Delete Outputs", "All output files and folders deleted successfully.")

    del_btn = tk.Button(root, text="Delete All Outputs", command=delete_outputs, fg='red')
    del_btn.pack(pady=8)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Start Pipeline", command=on_submit).pack(side='left', padx=10)
    tk.Button(btn_frame, text="Default", command=on_default).pack(side='left', padx=10)
    tk.Button(btn_frame, text="Restore Previous", command=on_restore).pack(side='left', padx=10)
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) / 2
    root.geometry(f"+{int(x)}+{int(y)}")
    root.mainloop()
    return ranges.get('participants'), ranges.get('questions'), selected_plots, plot_vars["save_stage_outputs"].get()

# ======================================================================================
# >>> ADDED: UTILITIES & CONFIG HELPERS
# ======================================================================================

def _parse_bool(s):
    return str(s).strip().lower() in ('1','true','yes','y','on')

def _coerce_num(x):
    return pd.to_numeric(x, errors='coerce')

def _overlay_for_part(config, part_label):
    """
    Read overlay coordinates from config.ini.
    Supports:
      [Overlay] shared across parts
      [Overlay.Part1] and [Overlay.Part2] (override per-part)
    Keys supported (normalized to 0..1 screen coords):
      question_x, question_y
      option_a_x, option_a_y, option_b_x, option_b_y, option_c_x, option_c_y, option_d_x, option_d_y
      timer_x, timer_y, submit_x, submit_y
      box_width, box_height (optional for drawing rectangles)
    """
    # defaults that match prior hard-coded values
    defaults = {
        'question_x': 0.05, 'question_y': 0.04,
        'option_a_x': 0.225, 'option_a_y': 0.625,
        'option_b_x': 0.625, 'option_b_y': 0.625,
        'option_c_x': 0.225, 'option_c_y': 0.825,
        'option_d_x': 0.625, 'option_d_y': 0.825,
        'timer_x'   : 0.92,  'timer_y'   : 0.08,
        'submit_x'  : 0.90,  'submit_y'  : 0.94,
        'box_width' : 0.20,  'box_height': 0.10
    }

    # Base section(s)
    overlay = dict(defaults)
    if 'Overlay' in config:
        for k, v in config['Overlay'].items():
            try:
                overlay[k.lower()] = float(v)
            except ValueError:
                pass

    # Part override section names
    part_key = 'Overlay.Part1' if part_label == 'Part 1' else ('Overlay.Part2' if part_label == 'Part 2' else None)
    if part_key and part_key in config:
        for k, v in config[part_key].items():
            try:
                overlay[k.lower()] = float(v)
            except ValueError:
                pass

    return overlay


def _get_option_coords(option_index, overlay, config=None):
    """Return (cx, cy) for the option at option_index (0-based).

    Prioritize explicit overlay keys (option_a_x, option_a_y, ...). If there are
    more options than explicit keys, compute coordinates using a simple grid
    determined by option_button_columns in config (default 2) and the known
    option row y-values in overlay.
    """
    # Map known explicit keys for first four options
    explicit_map = [
        ('option_a_x', 'option_a_y'),
        ('option_b_x', 'option_b_y'),
        ('option_c_x', 'option_c_y'),
        ('option_d_x', 'option_d_y')
    ]
    if option_index < len(explicit_map):
        xk, yk = explicit_map[option_index]
        return overlay.get(xk, 0.5), overlay.get(yk, 0.5)

    # Fallback: compute grid positions
    cols = 2
    if config is not None:
        try:
            cols = int(config.get('DIMENSIONS', 'option_button_columns', fallback=2))
        except Exception:
            cols = 2

    # Determine base x positions (left/right) from overlay or defaults
    left_x = overlay.get('option_a_x', 0.225)
    right_x = overlay.get('option_b_x', 0.625)
    x_positions = [left_x, right_x] if cols >= 2 else [overlay.get('option_a_x', 0.5)]

    # Determine row y positions using option_a_y and option_c_y as anchors
    y_top = overlay.get('option_a_y', 0.625)
    y_second = overlay.get('option_c_y', 0.825)
    row_height = (y_second - y_top) if (y_second is not None) else 0.18

    row = option_index // cols
    col = option_index % cols

    cx = x_positions[col] if col < len(x_positions) else x_positions[-1]
    cy = y_top + row * row_height
    return cx, cy

def _get_aoi_rectangles(config, part_label):
    """Returns a dictionary of AOI names to their bounding box coordinates (xmin, ymin, xmax, ymax)."""
    overlay = _overlay_for_part(config, part_label)
    rects = {}

    # Question AOI (assuming it's a text block, so we'll define a box around its top-left anchor)
    # We need to estimate width and height for the question text block. These are rough estimates.
    # A typical question might take up to 50% of the screen width and 20% of the height from its anchor.
    q_x, q_y = overlay.get('question_x', 0.05), overlay.get('question_y', 0.04)
    q_width, q_height = 0.70, 0.20 # Estimated width and height for the question area
    # Clip to [0,1]
    rects['Question'] = (
        max(0.0, q_x), max(0.0, q_y), min(1.0, q_x + q_width), min(1.0, q_y + q_height)
    )

    # Option AOIs (using box_width and box_height from config or defaults)
    bw, bh = overlay.get('box_width', 0.20), overlay.get('box_height', 0.10)
    for i in range(4): # Assuming up to 4 options (A, B, C, D)
        cx, cy = _get_option_coords(i, overlay, config)
        
        # Adjust cx, cy to be top-left corner for rectangle definition
        # The _get_option_coords returns center, so subtract half width/height
        xmin = cx - bw / 2
        ymin = cy - bh / 2
        xmax = cx + bw / 2
        ymax = cy + bh / 2

        # Apply specific adjustments for option positioning as done in visualize_heatmaps
        if i % 2 != 0:
            xmin -= bw * 0.2
            xmax -= bw * 0.2
        if i > 1:
            ymin += bh * 0.3
            ymax += bh * 0.3

        option_id = chr(ord('A') + i)
        # Clip each coordinate to the 0..1 range
        rects[f'Choice_{option_id}'] = (
            max(0.0, xmin), max(0.0, ymin), min(1.0, xmax), min(1.0, ymax)
        )

    # Timer AOI
    timer_x, timer_y = overlay.get('timer_x', 0.92), overlay.get('timer_y', 0.08)
    timer_width, timer_height = 0.08, 0.05 # Estimated size for timer display
    rects['Timer'] = (
        max(0.0, timer_x - timer_width/2), max(0.0, timer_y - timer_height/2),
        min(1.0, timer_x + timer_width/2), min(1.0, timer_y + timer_height/2)
    )

    # Submit AOI
    submit_x, submit_y = overlay.get('submit_x', 0.90), overlay.get('submit_y', 0.94)
    submit_width, submit_height = 0.10, 0.05 # Estimated size for submit button
    rects['Submit'] = (
        max(0.0, submit_x - submit_width/2), max(0.0, submit_y - submit_height/2),
        min(1.0, submit_x + submit_width/2), min(1.0, submit_y + submit_height/2)
    )

    return rects

# ======================================================================================
# STAGE 1: DATA CLEANING AND PREPARATION
# ======================================================================================

def clean_and_prepare_data(df, invalid_gaze_threshold, consecutive_zero_threshold=5, report_dir='reports'):
    """
    Cleans the raw data by handling zero/invalid values and calculates total 
    reading time per question for each participant.
    """
    # Use correct column names for gaze data
    gaze_x_col, gaze_y_col, timestamp_col = 'BPOGX', 'BPOGY', 'FPOGS'
    
    # Step 1: conservative sample-level validity using AND (avoid discarding noisy but usable samples)
    df['raw_validity_flag'] = df.get('BPOGV', 1) == 1

    # mark zero coordinates
    df['is_zero_coord'] = (df[gaze_x_col] == 0) & (df[gaze_y_col] == 0)

    # Detect consecutive zero runs per participant-question
    df = df.sort_values(by=['participant_id', 'question_id', timestamp_col])
    def _zero_run_len(x):
        # returns run length at each position (consecutive zeros)
        runs = (x != x.shift()).cumsum()
        sizes = x.groupby(runs).transform('size')
        return sizes

    # only compute on grouped is_zero_coord
    df['zero_run_len'] = df.groupby(['participant_id', 'question_id'])['is_zero_coord'].transform(_zero_run_len)

    # Consider a sample invalid only if validity flag false OR zero_run_len >= consecutive_zero_threshold
    df['raw_valid_gaze_sample'] = (df['raw_validity_flag']) & (~((df['is_zero_coord']) & (df['zero_run_len'] >= int(consecutive_zero_threshold))))

    # Calculate the number of invalid gaze samples per participant-question (for diagnostics)
    invalid_samples_count = df.groupby(['participant_id', 'question_id']).apply(
        lambda x: (~x['raw_valid_gaze_sample']).sum(), include_groups=False
    ).reset_index(name='invalid_gaze_count')

    total_samples_count = df.groupby(['participant_id', 'question_id']).size().reset_index(name='total_gaze_count')

    gaze_validity_stats = pd.merge(invalid_samples_count, total_samples_count, on=['participant_id', 'question_id'])
    gaze_validity_stats['invalid_gaze_ratio'] = gaze_validity_stats['invalid_gaze_count'] / gaze_validity_stats['total_gaze_count']

    # Ensure reports directory exists
    os.makedirs(report_dir, exist_ok=True)
    gaze_validity_csv = os.path.join(report_dir, 'gaze_validity_summary.csv')
    gaze_validity_stats.to_csv(gaze_validity_csv, index=False, float_format='%.4f')
    print(f"Saved gaze validity per trial summary to {gaze_validity_csv}")

    # Participant-level summary
    participant_summary = gaze_validity_stats.groupby('participant_id').agg(
        total_rows=('total_gaze_count', 'sum'),
        removed_rows=('invalid_gaze_count', 'sum')
    ).reset_index()
    participant_summary['valid_ratio'] = 1.0 - (participant_summary['removed_rows'] / participant_summary['total_rows']).replace({np.inf: 0, np.nan: 0})
    participant_csv = os.path.join(report_dir, 'participant_validity_summary.csv')
    participant_summary.to_csv(participant_csv, index=False, float_format='%.4f')
    print(f"Saved participant validity summary to {participant_csv}")

    # Identify participant-question pairs exceeding threshold but don't drop yet; return list for caller
    questions_to_remove = gaze_validity_stats[gaze_validity_stats['invalid_gaze_ratio'] > float(invalid_gaze_threshold)][['participant_id', 'question_id']]

    # Filter out high-invalid trials from further gaze-based onset/phase computations, but keep original for diagnostics
    if not questions_to_remove.empty:
        keys_to_remove_idx = pd.MultiIndex.from_frame(questions_to_remove)
        df_idx = pd.MultiIndex.from_frame(df[['participant_id', 'question_id']])
        valid_mask = ~df_idx.isin(keys_to_remove_idx)
        df_filtered = df[valid_mask].copy()
    else:
        df_filtered = df.copy()

    # Now keep only rows flagged as valid samples for timing calculations
    df_valid_samples = df_filtered[df_filtered['raw_valid_gaze_sample']].copy()
    df_valid_samples[timestamp_col] = pd.to_numeric(df_valid_samples[timestamp_col], errors='coerce')
    df_valid_samples.dropna(subset=[timestamp_col], inplace=True)
    df_valid_samples.sort_values(by=['participant_id', 'question_id', timestamp_col], inplace=True)

    # Calculate total time (t_ij) per question for each participant, preserving part info
    time_df = df_valid_samples.groupby(['participant_id', 'question_id', 'part'])[timestamp_col].agg(['min', 'max']).reset_index()
    time_df['t_ij'] = (time_df['max'] - time_df['min']) # Timestamp is already in seconds
    
    # valid_data flag + threshold (keep your logic)
    time_df['valid_data'] = time_df['t_ij'] > 0

    # Filter out questions with very short interaction time
    time_df = time_df[time_df['t_ij'] > 1.0] # Minimum 1 second interaction
    if time_df['t_ij'].max() > 300:
      print("⚠️ Warning: Some t_ij values exceed 300s — check timestamp units (FPOGS should be seconds).")

    # Save overall valid pair counts for logging
    total_pairs = time_df.shape[0]
    print(f"Total valid participant-question pairs after cleaning: {total_pairs}")

    # Return time_df and diagnostics for further reporting
    return time_df, gaze_validity_stats, participant_summary

# ======================================================================================
# >>> ADDED: TIME CAP SUPPORT (NO-TIMELIMIT/TIMELIMIT)
# ======================================================================================

def _apply_time_cap(df, time_cap_s):
    """Mark censored rows (t_ij > time_cap_s). If no cap, censored=False."""
    if time_cap_s is None:
        df['censored'] = False
        return df
    df['censored'] = df['t_ij'] > float(time_cap_s)
    return df

# ======================================================================================
# STAGE 2: OUTLIER DETECTION (INVALIDLY FAST ANSWERS)
# ======================================================================================

def detect_outliers(df, time_cap_s=None, exclude_censored=False, iqr_multiplier=1.5):
    """
    Identifies outliers based on reading time, considering different parts of the exam.
    Participants who answer too quickly are flagged as invalid for that specific question.
    Also computes per-question/part stats (Q1, median, Q3, IQR, LB, n_all).
    """
    # time cap flag
    df = _apply_time_cap(df.copy(), time_cap_s)

    # Calculate stats per question, separated by part
    base = df[~df['censored']] if (exclude_censored and 'censored' in df.columns) else df

    if base.empty:
        stats_per_question = pd.DataFrame(columns=['question_id','part','Q1','median','Q3','IQR','LB','n_all'])
    else:
        stats_per_question = base.groupby(['question_id', 'part'])['t_ij'].agg(
            Q1=lambda x: x.quantile(0.25),
            median='median',
            Q3=lambda x: x.quantile(0.75),
            n_all='count'
        ).reset_index()
        stats_per_question['IQR'] = stats_per_question['Q3'] - stats_per_question['Q1']
        stats_per_question['LB'] = stats_per_question['Q1'] - (iqr_multiplier * stats_per_question['IQR'])
    
    # Merge back on both question_id and part
    df = pd.merge(df, stats_per_question[['question_id', 'part', 'LB']], on=['question_id', 'part'], how='left')
    
    # Flag invalid times
    df['is_valid_time'] = np.where(df['LB'].notna(), df['t_ij'] >= df['LB'], False)
    df['invalid_time'] = ~df['is_valid_time']
    
    print(f"Identified {len(df[~df['is_valid_time']])} instances of invalidly fast answers.")
    
    return df, stats_per_question

# ======================================================================================
# STAGE 3: BEHAVIORAL LABELING (UP/NP)
# ======================================================================================

def apply_behavioral_labels(df, answer_data, time_cap_s=None, exclude_censored=False, iqr_multiplier=1.5):
    """
    Applies behavioral labels (UP/NP) based on the provided logic, considering
    only valid participants from the previous stage and separating by exam part.

    Computes correct-only stats (Q1_C, median_C, Q3_C, IQR_C, UF_C, n_correct_valid).
    label NA_no_correct when UF_C missing; keep INVALID for invalid_time rows.
    """
    # Merge with answer data to get correctness
    df = pd.merge(df, answer_data, on=['participant_id', 'question_id', 'part'])
    
    # Work only with valid-time rows
    valid_df = df[df['is_valid_time']].copy()

    # optionally remove censored rows before correct-only stats
    if exclude_censored and 'censored' in valid_df.columns:
        valid_for_correct = valid_df[~valid_df['censored']].copy()
    else:
        valid_for_correct = valid_df.copy()
    
    # Calculate stats for correct answers only, separated by part
    correct_answers_df = valid_for_correct[valid_for_correct['is_correct'] == 1]
    if correct_answers_df.empty:
        correct_stats = pd.DataFrame(columns=['question_id','part','Q1_C','median_C','Q3_C','IQR_C','UF_C','n_correct_valid'])
    else:
        correct_stats = correct_answers_df.groupby(['question_id', 'part'])['t_ij'].agg(
            Q1_C=lambda x: x.quantile(0.25),
            median_C='median',
            Q3_C=lambda x: x.quantile(0.75),
            n_correct_valid='count'
        ).reset_index()
        correct_stats['IQR_C'] = correct_stats['Q3_C'] - correct_stats['Q1_C']
        correct_stats['UF_C'] = correct_stats['Q3_C'] + (iqr_multiplier * correct_stats['IQR_C'])
    
    # Merge UF_C back into valid_df
    valid_df = pd.merge(valid_df, correct_stats[['question_id', 'part', 'UF_C']], on=['question_id', 'part'], how='left')
    
    # Define UP condition
    condition_up = (valid_df['is_correct'] == 0) | \
                   ((valid_df['is_correct'] == 1) & (valid_df['t_ij'] > valid_df['UF_C']))
    
    # NA_no_correct when UF_C is NaN (This occurs when there are no valid correct answers for a given question/part
    # to calculate UF_C, making it impossible to classify UP/NP based on correct answer thresholds.)
    no_correct_valid = valid_df['UF_C'].isna()
    valid_df['label'] = np.where(no_correct_valid, 'NA_no_correct',
                                 np.where(condition_up, 'UP', 'NP'))
    
    # Merge labels back into the original dataframe
    df = pd.merge(df, valid_df[['participant_id', 'question_id', 'part', 'label']], on=['participant_id', 'question_id', 'part'], how='left')
    df['label'] = df['label'].fillna('INVALID')
    
    return df, correct_stats

# ======================================================================================
# STAGE 4: FEATURE ENGINEERING (AOI ANALYSIS)
# ======================================================================================

def engineer_features(raw_df, processed_df):
    """
    Engineers features based on Areas of Interest (AOIs).
    Calculates time spent in major phases: 'Reading Question' vs 'Answering'.

    AOI-based phase onset from BKID (Question -> Choice_*),
    fallback to midpoint logic when AOI not available.
    """
    # Use correct column names
    timestamp_col = 'FPOGS'
    gaze_x_col, gaze_y_col = 'BPOGX', 'BPOGY'

    # We need the raw gaze data for this
    raw_df = raw_df.copy()
    raw_df[timestamp_col] = pd.to_numeric(raw_df[timestamp_col], errors='coerce')
    raw_df.dropna(subset=[timestamp_col], inplace=True)
    raw_df.sort_values(by=['participant_id', 'question_id', timestamp_col], inplace=True)

    # Determine AOIs dynamically based on config for each question/part
    def _assign_aoi(row, aoi_rects):
        x, y = row[gaze_x_col], row[gaze_y_col]
        for aoi_name, (xmin, ymin, xmax, ymax) in aoi_rects.items():
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return aoi_name
        return 'Other'

    # Apply AOI assignment per row, considering the part of the question
    # This requires iterating or grouping, which can be slow. Let's optimize.
    # First, get unique combinations of participant_id and question_id to get their part.
    unique_q_parts = raw_df[['participant_id', 'question_id', 'part']].drop_duplicates()
    
    aoi_assignments = []
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'))

    for _, row in unique_q_parts.iterrows():
        p_id, q_id, part = row['participant_id'], row['question_id'], row['part']
        aoi_rects = _get_aoi_rectangles(config, part)
        
        # Filter raw_df for current participant and question
        temp_df = raw_df[(raw_df['participant_id'] == p_id) & (raw_df['question_id'] == q_id)].copy()
        # Drop rows with missing gaze coordinates to avoid incorrect AOI 'Other' labels
        temp_df = temp_df.dropna(subset=[gaze_x_col, gaze_y_col])
        if not temp_df.empty:
            temp_df['AOI'] = temp_df.apply(lambda r: _assign_aoi(r, aoi_rects), axis=1)
            aoi_assignments.append(temp_df)

    if aoi_assignments:
        raw_df = pd.concat(aoi_assignments, ignore_index=True)
        print(f"--- AOI column created dynamically. Unique AOIs: {raw_df['AOI'].unique()}")
    else:
        print("Warning: No AOI assignments could be made. Setting AOI to 'Unknown'.")
        raw_df['AOI'] = 'Unknown'

    # compute onset based on AOI transition (Question -> Choice_*)
    def _phase_onset_by_aoi(g):
        g = g.sort_values(timestamp_col)
        prev = None
        for _, r in g.iterrows():
            aoi = r.get('AOI')
            if prev == 'Question' and isinstance(aoi, str) and aoi.startswith('Choice_'):
                return r[timestamp_col]
            prev = aoi
        return np.nan

    # use include_groups=False to avoid future warning
    onsets = (raw_df.groupby(['participant_id','question_id'])
          .apply(lambda g: _phase_onset_by_aoi(g), include_groups=False)
          .reset_index(name='phase_onset_s'))

    raw_df = pd.merge(raw_df, onsets, on=['participant_id', 'question_id'], how='left')

    # old midpoint logic as fallback
    trial_times = raw_df.groupby(['participant_id', 'question_id'])[timestamp_col].agg(['min', 'max']).reset_index()
    trial_times['midpoint'] = trial_times['min'] + (trial_times['max'] - trial_times['min']) / 2
    raw_df = pd.merge(raw_df, trial_times[['participant_id', 'question_id', 'midpoint']], on=['participant_id', 'question_id'])

    # choose AOI onset if exists else midpoint
    raw_df['effective_onset'] = np.where(raw_df['phase_onset_s'].notna(), raw_df['phase_onset_s'], raw_df['midpoint'])

    # phase by effective onset
    raw_df['phase'] = np.where(raw_df[timestamp_col] < raw_df['effective_onset'], 'Reading', 'Answering')

    # duration between samples
    raw_df['duration'] = raw_df.groupby(['participant_id', 'question_id'])[timestamp_col].diff().fillna(0)

    # Aggregate time spent in each phase
    phase_durations = raw_df.groupby(['participant_id', 'question_id', 'phase'])['duration'].sum().unstack(fill_value=0).reset_index()
    phase_durations.columns.name = None
    
    # Rename columns for clarity
    if 'Reading' in phase_durations.columns:
        phase_durations.rename(columns={'Reading': 'Reading_duration_s'}, inplace=True)
    if 'Answering' in phase_durations.columns:
        phase_durations.rename(columns={'Answering': 'Answering_duration_s'}, inplace=True)
    
    # aggregate AOI times too
    aoi_time = raw_df.groupby(['participant_id','question_id','AOI'])['duration'].sum().unstack(fill_value=0).reset_index()

    # Merge features into the main processed dataframe
    final_df = pd.merge(processed_df, phase_durations, on=['participant_id', 'question_id'], how='left')
    final_df = pd.merge(final_df, onsets, on=['participant_id', 'question_id'], how='left')  # keep onsets for inspection
    final_df = pd.merge(final_df, aoi_time, on=['participant_id','question_id'], how='left')
    
    return final_df


# ======================================================================================
# VISUALIZATION FUNCTIONS
# ======================================================================================

def visualize_stage1(df, ax):
    """Visualizes the distribution of time spent on questions, separated by exam part."""
    sns.boxplot(data=df, x='part', y='t_ij', ax=ax)
    ax.set_title('Stage 1: Distribution of Time per Question (t_ij)')
    ax.set_xlabel('Exam Part')
    ax.set_ylabel('Time (seconds)')
    ax.grid(True)


def visualize_stage2(df, ax):
    """Visualizes outlier detection results, separated by exam part."""
    # To avoid a cluttered plot, let's visualize a sample of questions
    sample_questions = sorted(df['question_id'].unique())[:5]
    sample_df = df[df['question_id'].isin(sample_questions)]
    
    sns.scatterplot(data=sample_df, x='question_id', y='t_ij', hue='is_valid_time', style='part', s=100, ax=ax)
    
    # Plot the lower bound line for reference
    if not sample_df.empty:
        lb_lines = sample_df[['question_id', 'part', 'LB']].drop_duplicates()
        sns.stripplot(data=lb_lines, x='question_id', y='LB', color='red', marker='_', ax=ax, size=15, jitter=False)

    ax.set_title('Stage 2: Outlier Detection (Invalidly Fast Answers)')
    ax.set_ylabel('Time (seconds)')
    ax.set_xlabel('Question ID')
    ax.legend(title='Status')
    ax.grid(True)


def visualize_stage3(df, ax):
    """Visualizes the distribution of behavioral labels, separated by exam part."""
    sns.countplot(data=df, x='label', hue='part', order=['NP', 'UP', 'INVALID', 'NA_no_correct'], ax=ax)
    ax.set_title('Stage 3: Distribution of Behavioral Labels')
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    ax.grid(axis='y')


def visualize_stage4(df, ax):
    """Visualizes the reading vs. answering phase durations, separated by exam part."""
    # Prepare data for boxplot, ensuring columns exist
    plot_cols = []
    if 'Reading_duration_s' in df.columns:
        plot_cols.append('Reading_duration_s')
    if 'Answering_duration_s' in df.columns:
        plot_cols.append('Answering_duration_s')
    
    if not plot_cols:
        print("Warning: Duration columns for visualization not found.")
        ax.text(0.5, 0.5, 'No data for Stage 4 plot', ha='center', va='center')
        ax.set_title('Stage 4: Reading vs. Answering Phase Durations')
        return

    plot_data = df[plot_cols + ['part']].melt(id_vars=['part'], var_name='Phase', value_name='Duration (s)')
    sns.boxplot(data=plot_data, x='Phase', y='Duration (s)', hue='part', ax=ax)
    ax.set_title('Stage 4: Reading vs. Answering Phase Durations')
    ax.grid(axis='y')


def visualize_heatmaps(df, viz_dir, question_texts, bg_image_part1=None, bg_image_part2=None, config=None, progress_queue=None, cancel_event=None):
    """Generates and saves heatmaps of gaze data with question text overlays + option IDs at precise overlay coords."""
    print("Generating gaze heatmaps for each participant and question...")
    if 'part' not in df.columns:
        print("Warning: 'part' column not found for heatmaps. Skipping part-specific organization.")
        return
    if cancel_event and cancel_event.is_set(): return

    participants = sorted(df['participant_id'].unique())
    total_images = sum(df[df['participant_id'] == pid]['question_id'].nunique() for pid in participants)
    image_counter = 0

    for participant_id in participants:
        participant_df = df[df['participant_id'] == participant_id]
        for q_id in sorted(participant_df['question_id'].unique()):
            if cancel_event and cancel_event.is_set(): return
            question_df = participant_df[participant_df['question_id'] == q_id]
            if question_df.empty or 'BPOGX' not in question_df.columns or 'BPOGY' not in question_df.columns:
                print(f"Warning: No gaze data for participant {participant_id}, question {q_id}. Skipping heatmap.")
                continue
            
            # Check for sufficient data points and variance to avoid kdeplot error
            if question_df.shape[0] < 2 or question_df['BPOGX'].nunique() < 2 or question_df['BPOGY'].nunique() < 2:
                print(f"Warning: Not enough data points or variance for participant {participant_id}, question {q_id}. Skipping heatmap.")
                continue
            
            part = question_df['part'].iloc[0] if pd.notna(question_df['part'].iloc[0]) else 'UnknownPart'
            part_str_for_path = part.replace(" ", "_")

            # Select the appropriate background image
            background_image = None
            if part == 'Part 1' and bg_image_part1 is not None:
                background_image = bg_image_part1
            elif part == 'Part 2' and bg_image_part2 is not None:
                background_image = bg_image_part2

            # Read overlay (positions/sizes) from config.ini
            overlay = _overlay_for_part(config, part) if config is not None else _overlay_for_part(configparser.ConfigParser(), part)

            fig, ax = plt.subplots(figsize=(12, 10))
            
            try:
                # Display the background image if it exists
                if background_image is not None:
                    # Flip the image vertically to correct for the inverted Y-axis used for gaze data
                    ax.imshow(np.flipud(background_image), extent=[0, 1, 0, 1], aspect='auto')

                # Add question and answer texts (content) and their IDs (from questions.json)
                q_data = question_texts.get(q_id)
                if q_data:
                    # Question Text
                    question_text = q_data['text']
                    wrapped_text = '\n'.join(textwrap.wrap(question_text, width=56))
                    
                    # Draw question text with contrasting stroke and light bbox so it's readable over any background
                    ax.text(overlay.get('question_x', 0.05), overlay.get('question_y', 0.04), wrapped_text,
                            ha='left', va='top', fontsize=16, color='white', family='sans-serif', zorder=5,
                            path_effects=[pe.withStroke(linewidth=3, foreground='black')],
                            bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))

                    # Option boxes & labels (IDs + text) at precise coords
                    bw, bh = overlay.get('box_width', 0.20), overlay.get('box_height', 0.10)

                    for i, option in enumerate(q_data['options']):
                        cx, cy = _get_option_coords(i, overlay, config)

                        # if col number is even
                        if i % 2 != 0:
                            cx = cx - bw * 0.2

                        # if is 2th option row
                        if i>1:
                            cy = cy + bh * 0.3
                            
                        ax.text(cx, cy - bh*0.3, option.get('id', ''), ha='center', va='center',
                                fontsize=12, color='white', family='sans-serif', zorder=5, fontweight='bold',
                                path_effects=[pe.withStroke(linewidth=3, foreground='black')],
                                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))

                        ax.text(cx, cy + bh*0.3, option.get('text', ''), ha='center', va='center',
                                fontsize=12, color='white', family='sans-serif', zorder=5,
                                path_effects=[pe.withStroke(linewidth=3, foreground='black')],
                                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))

                # Filter gaze data to only include points within the main box (full screen is 0..1)
                main_box_xmin = 0.1
                main_box_xmax = 0.9
                main_box_ymin = 0.1
                main_box_ymax = 0.9

                # Filter gaze data to only include points within the main box
                filtered_gaze = question_df[
                    (question_df['BPOGX'] >= main_box_xmin) & (question_df['BPOGX'] <= main_box_xmax) &
                    (question_df['BPOGY'] >= main_box_ymin) & (question_df['BPOGY'] <= main_box_ymax)
                ]

                # English: Draw heatmap only if there is enough diverse data
                if filtered_gaze.shape[0] >= 2 and filtered_gaze['BPOGX'].nunique() >= 2 and filtered_gaze['BPOGY'].nunique() >= 2:
                    dynamic_levels = min(10, max(3, filtered_gaze.shape[0] // 10))
                    try:
                        sns.kdeplot(
                            x=filtered_gaze['BPOGX'],
                            y=filtered_gaze['BPOGY'],
                            ax=ax, fill=True, cmap='viridis',
                            #levels=dynamic_levels, alpha=0.6, zorder=2 # Number of graph layers
                            levels=40, alpha=0.6, zorder=2 # Number of graph layers
                        )
                        plt.colorbar(ax.collections[0], ax=ax, orientation='horizontal', label='Density')
                    except Exception as e:
                        print(f"ERROR: Could not generate heatmap for participant {participant_id}, question {q_id} due to: {e}. Only background and boxes will be shown.")
                else:
                    print(f"Warning: Not enough filtered gaze data for participant {participant_id}, question {q_id}. Only background and boxes will be shown.")
                
                ax.set_title(f'Gaze Heatmap for {participant_id} | {part} | {q_id}')
                ax.set_xlabel('Gaze X Coordinate')
                ax.set_ylabel('Gaze Y Coordinate')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.invert_yaxis() # Match gaze data (0,0 at top-left)
                ax.grid(False)
                ax.set_aspect('auto', adjustable='box')

                # Save in a subfolder for each participant and part
                participant_viz_dir = os.path.join(viz_dir, 'heatmaps', f'participant_{participant_id}')
                part_viz_dir = os.path.join(participant_viz_dir, part_str_for_path)
                os.makedirs(part_viz_dir, exist_ok=True)
                heatmap_path = os.path.join(part_viz_dir, f'gaze_heatmap_{q_id}.png')
                plt.savefig(heatmap_path)
                print(f"Saved heatmap for participant {participant_id}, question {q_id} ({part}) to {heatmap_path}")
            except ValueError as e:
                print(f"ERROR: Could not generate heatmap for participant {participant_id}, question {q_id} due to plotting error: {e}. Skipping.")
            finally:
                plt.close(fig)
                image_counter += 1
                if progress_queue is not None and total_images > 0:
                    percent = int(100 * image_counter / total_images)
                    progress_queue.put(("stage_progress", (percent, f"Heatmap: {participant_id}, {q_id} ({image_counter}/{total_images})")))


def visualize_scatterplots(df, viz_dir, question_texts, bg_image_part1=None, bg_image_part2=None, config=None, progress_queue=None, cancel_event=None):
    """Generates and saves scatter plots of gaze points for each participant and question.

    Saves images under: <viz_dir>/gaze_scatterplots_<participant_id>/<Part_1_or_2>/gaze_scatter_Q*.png
    Uses same overlay and background logic as heatmaps so both visuals align.
    """
    print("Generating gaze scatter plots for each participant and question...")
    if 'part' not in df.columns:
        print("Warning: 'part' column not found for scatter plots. Skipping part-specific organization.")
        return
    if cancel_event and cancel_event.is_set(): return

    participants = sorted(df['participant_id'].unique())
    total_images = sum(df[df['participant_id'] == pid]['question_id'].nunique() for pid in participants)
    image_counter = 0

    for participant_id in participants:
        participant_df = df[df['participant_id'] == participant_id]
        for q_id in sorted(participant_df['question_id'].unique()):
            if cancel_event and cancel_event.is_set(): return
            question_df = participant_df[participant_df['question_id'] == q_id]
            if question_df.empty or 'BPOGX' not in question_df.columns or 'BPOGY' not in question_df.columns:
                print(f"Warning: No gaze data for participant {participant_id}, question {q_id}. Skipping scatter plot.")
                continue

            part = question_df['part'].iloc[0] if pd.notna(question_df['part'].iloc[0]) else 'UnknownPart'
            part_str_for_path = part.replace(" ", "_")

            # Select background
            background_image = None
            if part == 'Part 1' and bg_image_part1 is not None:
                background_image = bg_image_part1
            elif part == 'Part 2' and bg_image_part2 is not None:
                background_image = bg_image_part2

            # Read overlay
            overlay = _overlay_for_part(config, part) if config is not None else _overlay_for_part(configparser.ConfigParser(), part)

            main_box_xmin = 0.04
            main_box_xmax = 0.96
            main_box_ymin = 0.04
            main_box_ymax = 0.96

            # Filter gaze inside screen
            filtered_gaze = question_df[ (question_df['BPOGX'] >= main_box_xmin) & (question_df['BPOGX'] <= main_box_xmax) &
                                         (question_df['BPOGY'] >= main_box_ymin) & (question_df['BPOGY'] <= main_box_ymax) ]

            if filtered_gaze.shape[0] == 0:
                print(f"Warning: No valid gaze points for participant {participant_id}, question {q_id}. Skipping scatter plot.")
                continue

            fig, ax = plt.subplots(figsize=(10, 7))
            try:
                if background_image is not None:
                    ax.imshow(np.flipud(background_image), extent=[0, 1, 0, 1], aspect='auto')

                # Draw overlays (question text and option boxes) similar to heatmaps
                q_data = question_texts.get(q_id)
                if q_data:
                    question_text = q_data['text']
                    wrapped_text = '\n'.join(textwrap.wrap(question_text, width=52))
                    ax.text(overlay.get('question_x', 0.05), overlay.get('question_y', 0.04), wrapped_text,
                            ha='left', va='top', fontsize=14, color='white', family='sans-serif', zorder=5,
                            path_effects=[pe.withStroke(linewidth=3, foreground='black')],
                            bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))

                    bw, bh = overlay.get('box_width', 0.20), overlay.get('box_height', 0.10)
                    # Draw all options (works for 4 or 5 options)
                    for i, option in enumerate(q_data['options']):
                        cx, cy = _get_option_coords(i, overlay, config)

                        # if col number is even
                        # Shift the x-coordinate slightly to the left for better alignment
                        if i % 2 != 0:
                            cx = cx - bw * 0.2

                        # if is 2th option row
                        # Shift the y-coordinate slightly down for better alignment
                        if i>1:
                            cy = cy + bh * 0.3

                        # Draw option ID and text
                        ax.text(cx, cy - bh*0.3, option.get('id', ''), ha='center', va='center',
                                fontsize=12, color='white', family='sans-serif', zorder=5, fontweight='bold',
                                path_effects=[pe.withStroke(linewidth=3, foreground='black')],
                                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))

                        ax.text(cx, cy + bh*0.3, option.get('text', ''), ha='center', va='center',
                                fontsize=12, color='white', family='sans-serif', zorder=5,
                                path_effects=[pe.withStroke(linewidth=3, foreground='black')],
                                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))

                # Scatter the gaze points
                ax.scatter(filtered_gaze['BPOGX'], filtered_gaze['BPOGY'], s=30, c='red', alpha=0.6, zorder=4)
                ax.set_title(f'Gaze Scatter for {participant_id} | {part} | {q_id}')
                ax.set_xlabel('Gaze X Coordinate')
                ax.set_ylabel('Gaze Y Coordinate')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.invert_yaxis()
                ax.grid(False)

                part_viz_dir = os.path.join(viz_dir, 'points', f'participant_{participant_id}', part_str_for_path)
                os.makedirs(part_viz_dir, exist_ok=True)
                scatter_path = os.path.join(part_viz_dir, f'gaze_point_{q_id}.png')
                plt.savefig(scatter_path)
                print(f"Saved scatter plot for participant {participant_id}, question {q_id} ({part}) to {scatter_path}")
            except Exception as e:
                print(f"ERROR: Could not generate scatter plot for participant {participant_id}, question {q_id}: {e}")
            finally:
                plt.close(fig)
                image_counter += 1
                if progress_queue is not None and total_images > 0:
                    percent = int(100 * image_counter / total_images)
                    progress_queue.put(("stage_progress", (percent, f"Scatter: {participant_id}, {q_id} ({image_counter}/{total_images})")))


def visualize_aoi_summary_per_question(df, correct_answers, viz_dir, progress_queue, cancel_event=None):
    """
    Generates a bar chart showing the average time participants spent on different AOIs for each question.
    The AOIs are 'Question', 'Correct Answer', and 'Other Answers'.
    This function now correctly handles per-participant randomized answer choices.
    """
    print("Generating AOI summary per question bar chart...")
    if cancel_event and cancel_event.is_set(): return

    aoi_cols = [col for col in df.columns if col.startswith('Choice_') or col == 'Question']
    if not aoi_cols or 'Question' not in aoi_cols:
        print("Warning: AOI columns ('Question', 'Choice_*') not found for 'AOI Summary' visualization.")
        return

    try:
        progress_queue.put(("stage_progress", (10, "Preparing data for AOI summary chart...")))

        # Build per-participant correct-letter mapping from question_exams files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        qexam_dir = os.path.join(script_dir, 'question_exams')
        participant_map = {}
        if os.path.exists(qexam_dir):
            for fname in os.listdir(qexam_dir):
                if not fname.lower().endswith('.json'):
                    continue
                try:
                    base = os.path.splitext(fname)[0]
                    num = base.split('_')[-1]
                    participant_id = f'participant_{num}'
                    fpath = os.path.join(qexam_dir, fname)
                    with open(fpath, 'r', encoding='utf-8') as jf:
                        qdata = json.load(jf)
                        mapping = {}
                        for part in ('Part1', 'Part2'):
                            for item in qdata.get(part, []):
                                qid = item.get('question_id')
                                opts = item.get('options', [])
                                for idx, opt in enumerate(opts):
                                    oid = opt.get('id', '')
                                    if isinstance(oid, str) and oid.endswith('-C'):
                                        letter = chr(ord('A') + idx)
                                        mapping[f'Q{qid}'] = letter
                                        break
                        if mapping:
                            participant_map[participant_id] = mapping
                except Exception as e:
                    print(f"Warning: Could not process exam file {fname}: {e}")

        # Group by participant and question to compute mean AOI times per trial
        per_trial = df[['participant_id', 'question_id'] + aoi_cols].groupby(['participant_id', 'question_id']).mean().reset_index()

        per_trial_rows = []
        for _, prow in per_trial.iterrows():
            pid = prow['participant_id']
            qid = prow['question_id']
            
            correct_letter = participant_map.get(pid, {}).get(qid)

            row = {'participant_id': pid, 'question_id': qid}
            row['Question'] = prow.get('Question', 0.0)
            
            if correct_letter:
                correct_col = f'Choice_{correct_letter}'
                row['Correct_Answer'] = prow.get(correct_col, 0.0)
                
                other_sum = 0.0
                for letter in ['A', 'B', 'C', 'D']:
                    if letter != correct_letter:
                        choice_col = f'Choice_{letter}'
                        other_sum += prow.get(choice_col, 0.0)
                row['Other_Answers'] = other_sum
            else:
                print(f"Warning: Correct answer not found for participant {pid}, question {qid}. Summing all choices into 'Other_Answers'.")
                row['Correct_Answer'] = 0.0
                other_sum = 0.0
                for letter in ['A', 'B', 'C', 'D']:
                    choice_col = f'Choice_{letter}'
                    other_sum += prow.get(choice_col, 0.0)
                row['Other_Answers'] = other_sum
            
            per_trial_rows.append(row)

        if not per_trial_rows:
            print("Warning: No data to generate AOI summary chart after processing per-participant answers.")
            return

        per_trial_df = pd.DataFrame(per_trial_rows)

        # Now average across participants for each question
        avg_times = per_trial_df.groupby('question_id')[['Question', 'Correct_Answer', 'Other_Answers']].mean().reset_index()

        if cancel_event and cancel_event.is_set(): return
        progress_queue.put(("stage_progress", (50, "Creating AOI summary visualization...")))

        # Prepare pivot table for seconds
        avg_times = avg_times.set_index('question_id')
        for c in ['Question', 'Correct_Answer', 'Other_Answers']:
            if c not in avg_times.columns:
                avg_times[c] = 0.0

        pivot_df = avg_times[['Question', 'Correct_Answer', 'Other_Answers']].copy()

        # compute percent breakdown per question (row-wise)
        row_sums = pivot_df.sum(axis=1)
        percent_df = pivot_df.div(row_sums.replace({0: np.nan}), axis=0).fillna(0.0) * 100

        # Reorder rows by question id for plotting
        pivot_df.sort_index(inplace=True)
        percent_df = percent_df.loc[pivot_df.index]

        # Build human-readable multi-line x tick labels
        try:
            summary_labels = []
            for q in pivot_df.index:
                secs_q = float(pivot_df.at[q, 'Question'])
                secs_c = float(pivot_df.at[q, 'Correct_Answer'])
                secs_o = float(pivot_df.at[q, 'Other_Answers'])
                pct_q = float(percent_df.at[q, 'Question'])
                pct_c = float(percent_df.at[q, 'Correct_Answer'])
                pct_o = float(percent_df.at[q, 'Other_Answers'])
                lbl = (f"{q}\n"
                       f"Q: {pct_q:.1f}% ({secs_q:.1f}s)\n"
                       f"C: {pct_c:.1f}% ({secs_c:.1f}s)\n"
                       f"O: {pct_o:.1f}% ({secs_o:.1f}s)")
                summary_labels.append(lbl)
        except Exception:
            summary_labels = list(pivot_df.index.astype(str))

        # Colors
        colors = {'Question': 'skyblue', 'Correct_Answer': 'mediumseagreen', 'Other_Answers': 'lightcoral'}

        # Create a two-panel stacked bar: top = seconds, bottom = percent
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Plot seconds (stacked)
        pivot_df.plot(kind='bar', stacked=True, ax=ax1, color=[colors.get(c) for c in pivot_df.columns])
        ax1.set_title('Average Time Spent on AOIs per Question (Stacked seconds)')
        ax1.set_ylabel('Average Time (seconds)')
        ax1.legend(title='Area of Interest', bbox_to_anchor=(1.02, 1), loc='upper left')

        # Annotate seconds values
        try:
            positions = np.arange(len(pivot_df))
            bottoms = np.zeros(len(pivot_df))
            for col in pivot_df.columns:
                heights = pivot_df[col].values
                for i, h in enumerate(heights):
                    if h > 0.0 and h >= (pivot_df.sum(axis=1).max() * 0.02):
                        ax1.text(positions[i], bottoms[i] + h / 2, f"{h:.1f}s", ha='center', va='center', fontsize=8, color='black')
                    bottoms[i] += h
        except Exception:
            pass

        # percent plot
        percent_df.plot(kind='bar', stacked=True, ax=ax2, color=[colors.get(c) for c in percent_df.columns])
        ax2.set_title('Relative Attention Distribution per Question (Percent)')
        ax2.set_ylabel('Percent (%)')
        ax2.set_xlabel('Question ID')
        ax2.legend(title='Area of Interest', bbox_to_anchor=(1.02, 1), loc='upper left')

        # Apply multi-line labels
        try:
            ax2.set_xticklabels(summary_labels, rotation=0, ha='center', fontsize=8)
        except Exception:
            plt.xticks(rotation=45, ha='right')

        plt.subplots_adjust(bottom=0.30)
        plt.tight_layout()

        # Annotate percent values
        try:
            positions = np.arange(len(percent_df))
            bottoms_p = np.zeros(len(percent_df))
            for col in percent_df.columns:
                vals = percent_df[col].values
                for i, v in enumerate(vals):
                    if v >= 3.0:
                        ax2.text(positions[i], bottoms_p[i] + v / 2, f"{v:.1f}%", ha='center', va='center', fontsize=8, color='black')
                    bottoms_p[i] += v
        except Exception:
            pass

        # Save plot and CSV
        output_path = os.path.join(viz_dir, 'aoi_summary_per_question.png')
        plt.savefig(output_path)
        print(f"Saved AOI summary per question (seconds + percent) to {output_path}")
        
        try:
            summary_df = pivot_df.copy()
            summary_df['Question_pct'] = percent_df['Question']
            summary_df['Correct_Answer_pct'] = percent_df['Correct_Answer']
            summary_df['Other_Answers_pct'] = percent_df['Other_Answers']
            csv_path = os.path.join(viz_dir, 'avg_aoi_per_question.csv')
            summary_df.reset_index().rename(columns={
                'question_id': 'question_id',
                'Question': 'Question_s',
                'Correct_Answer': 'Correct_Answer_s',
                'Other_Answers': 'Other_Answers_s'
            }).to_csv(csv_path, index=False, float_format='%.3f')
            print(f"Saved AOI per-question summary CSV to {csv_path}")
        except Exception as e:
            print(f"Could not save AOI summary CSV: {e}")

        progress_queue.put(("stage_progress", (100, "AOI summary chart generated.")))

    except Exception as e:
        print(f"Error generating AOI summary chart: {e}")
    finally:
        plt.close()


def visualize_aoi_time_per_question(df, viz_dir, question_texts, progress_queue, cancel_event=None):
    """Generates bar charts for average time spent in each AOI per question."""
    print("Generating AOI time per question bar charts...")
    if 'aoi_cols' not in df.columns and not any(col.startswith('Choice_') for col in df.columns):
        print("Warning: No AOI columns found for 'AOI Time per Question' visualization.")
        return
    if cancel_event and cancel_event.is_set(): return

    aoi_cols = [col for col in df.columns if col.startswith('Choice_') or col in ['Question', 'Timer', 'Submit']]
    if not aoi_cols:
        print("Warning: No AOI columns found for 'AOI Time per Question' visualization.")
        return

    try:
        progress_queue.put(("stage_progress", (20, "Calculating average AOI time per question...")))
        if cancel_event and cancel_event.is_set(): return
        # Calculate average time per AOI for each question
        aoi_avg_time_q = df.groupby('question_id')[aoi_cols].mean().reset_index()
        aoi_avg_time_q_melted = aoi_avg_time_q.melt(id_vars='question_id', var_name='AOI', value_name='Average Duration (s)')

        progress_queue.put(("stage_progress", (50, "Creating visualization...")))
        if cancel_event and cancel_event.is_set(): return
        
        plt.figure(figsize=(15, 8))
        sns.barplot(data=aoi_avg_time_q_melted, x='question_id', y='Average Duration (s)', hue='AOI', palette='viridis')
        plt.title('Average Time Spent in Each AOI per Question')
        plt.xlabel('Question ID')
        plt.ylabel('Average Duration (s)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='AOI', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        aoi_q_path = os.path.join(viz_dir, 'aoi_time_per_question.png')
        plt.savefig(aoi_q_path)
        print(f"Saved AOI time per question bar chart to {aoi_q_path}")
        progress_queue.put(("stage_progress", (100, "AOI time per question chart generated.")))
    finally:
        plt.close()
        print("--- AOI time per question chart generation finished ---")

def visualize_aoi_time_per_label(df, viz_dir, progress_queue, cancel_event=None):
    """Generates bar charts for average time spent in each AOI per behavioral label (UP/NP)."""
    print("Generating AOI time per label bar charts...")
    if 'aoi_cols' not in df.columns and not any(col.startswith('Choice_') for col in df.columns):
        print("Warning: No AOI columns found for 'AOI Time per Label' visualization.")
        return
    if cancel_event and cancel_event.is_set(): return

    aoi_cols = [col for col in df.columns if col.startswith('Choice_') or col in ['Question', 'Timer', 'Submit']]
    if not aoi_cols:
        print("Warning: No AOI columns found for 'AOI Time per Label' visualization.")
        return

    # Filter out 'INVALID' and 'NA_no_correct' labels for this analysis if they exist
    filtered_df = df[df['label'].isin(['UP', 'NP'])].copy()
    if filtered_df.empty:
        print("Warning: No valid UP/NP labels found for 'AOI Time per Label' visualization.")
        return

    try:
        progress_queue.put(("stage_progress", (20, "Calculating average AOI time per label...")))
        if cancel_event and cancel_event.is_set(): return
        # Calculate average time per AOI for each label
        aoi_avg_time_l = filtered_df.groupby('label')[aoi_cols].mean().reset_index()
        aoi_avg_time_l_melted = aoi_avg_time_l.melt(id_vars='label', var_name='AOI', value_name='Average Duration (s)')

        progress_queue.put(("stage_progress", (70, "Creating bar chart for AOI time per label...")))
        if cancel_event and cancel_event.is_set(): return
        plt.figure(figsize=(12, 7))
        sns.barplot(data=aoi_avg_time_l_melted, x='label', y='Average Duration (s)', hue='AOI', palette='viridis')
        plt.title('Average Time Spent in Each AOI per Behavioral Label (UP/NP)')
        plt.xlabel('Behavioral Label')
        plt.ylabel('Average Duration (s)')
        plt.legend(title='AOI', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        aoi_l_path = os.path.join(viz_dir, 'aoi_time_per_label.png')
        plt.savefig(aoi_l_path)
        print(f"Saved AOI time per label bar chart to {aoi_l_path}")
        progress_queue.put(("stage_progress", (100, "AOI time per label chart generated.")))
    finally:
        plt.close()

# ======================================================================================
# REPORT (MARKDOWN) GENERATOR
# ======================================================================================

def _df_to_markdown_table(df, max_rows=20):
    df_show = df.copy()
    if len(df_show) > max_rows:
        df_show = df_show.head(max_rows)
    return df_show.to_markdown(index=False)

def write_html_report(report_path, stats_all, stats_c, labeled_df, final_df, summary_img_path, viz_dir, config=None):
    """
    Creates a structured HTML report describing pipeline stages and results,
    embedding images and tables.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Label distribution
    label_counts = labeled_df['label'].value_counts(dropna=False).rename_axis('label').reset_index(name='count')

    # Basic coverage
    n_participants = labeled_df['participant_id'].nunique()
    n_questions = labeled_df['question_id'].nunique()

    html_content = []
    html_content.append("<!DOCTYPE html>")
    html_content.append("<html lang=\"en\">")
    html_content.append("<head>")
    html_content.append("    <meta charset=\"UTF-8\">")
    html_content.append("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">")
    html_content.append("    <title>Data Mining Pipeline Report</title>")
    html_content.append("    <link rel=\"stylesheet\" href=\"https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css\">")
    html_content.append("    <style>")
    html_content.append("        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; background-color: #f4f4f4; color: #333; }")
    html_content.append("        .container { background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }")
    html_content.append("        h1, h2, h3 { color: #0056b3; }")
    html_content.append("        img { max-width: 100%; height: auto; display: block; margin: 15px 0; border: 1px solid #ddd; border-radius: 4px; cursor: zoom-in; }")
    html_content.append("        table { width: 100%; border-collapse: collapse; margin: 15px 0; }")
    html_content.append("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
    html_content.append("        th { background-color: #f2f2f2; }")
    html_content.append("        .section-divider { border-top: 2px solid #eee; margin: 30px 0; }")
    html_content.append("        /* Modal for image zoom */")
    html_content.append("        .modal {")
    html_content.append("            display: none; /* Hidden by default */")
    html_content.append("            position: fixed; /* Stay in place */")
    html_content.append("            z-index: 1000; /* Sit on top */")
    html_content.append("            padding-top: 50px; /* Location of the box */")
    html_content.append("            left: 0;")
    html_content.append("            top: 0;")
    html_content.append("            width: 100%; /* Full width */")
    html_content.append("            height: 100%; /* Full height */")
    html_content.append("            overflow: auto; /* Enable scroll if needed */")
    html_content.append("            background-color: rgb(0,0,0); /* Fallback color */")
    html_content.append("            background-color: rgba(0,0,0,0.9); /* Black w/ opacity */")
    html_content.append("        }")
    html_content.append("        .modal-content {")
    html_content.append("            margin: auto;")
    html_content.append("            display: block;")
    html_content.append("            width: 80%;")
    html_content.append("            max-width: 1200px;")
    html_content.append("        }")
    html_content.append("        .modal-content, #caption {")
    html_content.append("            animation-name: zoom;")
    html_content.append("            animation-duration: 0.6s;")
    html_content.append("        }")
    html_content.append("        @keyframes zoom {")
    html_content.append("            from {transform:scale(0)}")
    html_content.append("            to {transform:scale(1)}")
    html_content.append("        }")
    html_content.append("        .close {")
    html_content.append("            position: absolute;")
    html_content.append("            top: 15px;")
    html_content.append("            right: 35px;")
    html_content.append("            color: #f1f1f1;")
    html_content.append("            font-size: 40px;")
    html_content.append("            font-weight: bold;")
    html_content.append("            transition: 0.3s;")
    html_content.append("        }")
    html_content.append("        .close:hover,")
    html_content.append("        .close:focus {")
    html_content.append("            color: #bbb;")
    html_content.append("            text-decoration: none;")
    html_content.append("            cursor: pointer;")
    html_content.append("        }")
    html_content.append("        #caption {")
    html_content.append("            margin: auto;")
    html_content.append("            display: block;")
    html_content.append("            width: 80%;")
    html_content.append("            max-width: 700px;")
    html_content.append("            text-align: center;")
    html_content.append("            color: #ccc;")
    html_content.append("            padding: 10px 0;")
    html_content.append("            height: 150px;")
    html_content.append("        }")
    html_content.append("    </style>")
    html_content.append("</head>")
    html_content.append("<body>")
    html_content.append("    <div class=\"container\">")
    html_content.append("        <h1 class=\"mb-4\">Data Mining Pipeline Report</h1>")
    html_content.append("        <p>This report presents a comprehensive analysis of eye-tracking data, processed through a multi-stage pipeline designed to extract meaningful behavioral insights. The pipeline systematically cleans raw gaze data, identifies outliers, labels participant responses, and engineers features related to Areas of Interest (AOIs) and cognitive phases. The objective is to provide a robust framework for understanding user interaction patterns in response to presented stimuli, suitable for academic research and publication.</p>")
    html_content.append("        <p>The analysis covers data from a total of <strong>{n_participants} unique participants</strong> and <strong>{n_questions} unique questions</strong>. Each stage of the pipeline is detailed below, including the methodologies applied, key variables computed, and the rationale behind the processing steps.</p>")

    html_content.append("        <h2 class=\"mt-5\">Overview</h2>")
    html_content.append(f"        <p><strong>Unique participants:</strong> {n_participants}</p>")
    html_content.append(f"        <p><strong>Unique questions:</strong> {n_questions}</p>")
    # if summary_img_path and os.path.exists(summary_img_path):
    #     summary_img_rel_path = os.path.relpath(summary_img_path, os.path.dirname(report_path)).replace('\\', '/')
    #     html_content.append(f"        <img src=\"{summary_img_rel_path}\" alt=\"Pipeline Summary Image\" class=\"img-fluid\">")
    # else:
    #     html_content.append("        <p><em>Pipeline Summary Image not found.</em></p>")
    html_content.append("        <div class=\"section-divider\"></div>")

    html_content.append("        <h2 class=\"mt-5\">Stage 1 — Data Cleaning and Interaction Time (t_ij) Computation</h2>")
    html_content.append("        <p><strong>Objective:</strong> This initial stage focuses on refining raw eye-tracking data by removing erroneous gaze samples and calculating the total interaction time for each participant-question pair.</p>")
    html_content.append("<li>Invalid samples are removed only when the validity flag (BPOGV) is 0 <b>and</b> when long consecutive zero sequences in gaze coordinates (BPOGX, BPOGY) occur. "
            "Isolated single zeros are tolerated and not removed. This reflects the new AND logic with consecutive-zero detection.</li>")
    html_content.append("        <ul>")
    html_content.append("            <li><strong>Invalid Gaze Sample Removal:</strong> Gaze samples are considered invalid and subsequently removed if their `BPOGV` (Binocular Point of Gaze Validity) value is not equal to 1, or if their gaze coordinates (`BPOGX`, `BPOGY`) are precisely (0,0). These conditions typically indicate data loss or tracking errors.</li>")
    html_content.append("            <li><strong>Interaction Time (t_ij) Computation:</strong> For each unique combination of participant, question, and exam part, the total interaction duration, denoted as <strong>t_ij</strong>, is calculated. This metric represents the cumulative time a participant spent viewing a specific question. Following this, interactions shorter than 1 second are removed, as they are considered too brief to represent meaningful engagement.</li>")
    html_content.append("        </ul>")
    if 't_ij' in final_df.columns:
        html_content.append(f"        <p><strong>Summary Statistics for t_ij:</strong> Mean = {final_df['t_ij'].mean():.2f}s, Median = {final_df['t_ij'].median():.2f}s, Standard Deviation = {final_df['t_ij'].std():.2f}s</p>")
    html_content.append("        <div class=\"section-divider\"></div>")

    html_content.append("        <h2 class=\"mt-5\">Stage 2 — Fast Outlier Detection (Lower Bound - LB)</h2>")
    html_content.append("        <p><strong>Objective:</strong> This stage identifies and flags unusually short interaction times (t_ij) that may represent superficial engagement or premature responses, using a statistical lower bound (LB) threshold.</p>")
    html_content.append("        <h3 class=\"mt-4\">Methodology:</h3>")
    html_content.append("        <ul>")
    html_content.append("            <li><strong>Quartile and Interquartile Range (IQR) Computation:</strong> For each unique question and exam part, the first quartile (Q1), median, third quartile (Q3), and Interquartile Range (IQR = Q3 - Q1) of the `t_ij` values are calculated. These statistics provide a robust measure of the central tendency and spread of interaction times, minimizing the influence of extreme values.</li>")
    html_content.append("            <li><strong>Lower Bound (LB) Calculation:</strong> The Lower Bound (LB) is computed as $Q1 - 1.5 \times IQR$. This formula is a standard method for identifying potential outliers in a dataset, where values falling below the LB are considered statistically anomalous.</li>")
    html_content.append("            <li><strong>Time Validity Flagging:</strong> An interaction is flagged as <strong>`invalid_time`</strong> if its `t_ij` value is less than the calculated `LB` for that specific question and part. This identifies interactions that are significantly shorter than the typical engagement duration.</li>")
    html_content.append("        </ul>")
    if not stats_all.empty:
        html_content.append("        <h3 class=\"mt-4\">Sample of Computed Thresholds (LB)</h3>")
        html_content.append("        <p>The table below shows a sample of the calculated Q1, Median, Q3, IQR, and LB values for different question-part combinations. These thresholds are crucial for identifying outliers in interaction times.</p>")
        html_content.append(stats_all.head(20).to_html(index=False, classes='table table-striped table-bordered'))
    html_content.append("        <div class=\"section-divider\"></div>")

    html_content.append("        <h2 class=\"mt-5\">Stage 3 — Behavioral Labeling (Unusual/Normal Performance - UP/NP)</h2>")
    html_content.append("        <p><strong>Objective:</strong> This stage assigns behavioral labels (Unusual Performance - UP, Normal Performance - NP, Invalid, or Not Applicable) to each participant's response based on their correctness and interaction time relative to a statistically derived upper fence.</p>")
    html_content.append("        <h3 class=\"mt-4\">Methodology:</h3>")
    html_content.append("        <ul>")
    html_content.append("            <li><strong>Filtering for Valid Records:</strong> Labeling is performed exclusively on records deemed valid from Stage 2 (i.e., not flagged as `invalid_time`).</li>")
    html_content.append("            <li><strong>Correct Answer Statistics (Q1_C, median_C, Q3_C, IQR_C):</strong> Similar to Stage 2, quartile and IQR values are computed, but specifically for `t_ij` values associated with <strong>only valid correct answers</strong> for each question and part. This creates a baseline for efficient, correct responses.</li>")
    html_content.append("            <li><strong>Upper Fence for Correct Answers (UF_C):</strong> The Upper Fence for Correct answers (UF_C) is calculated as $Q3_C + 1.5 \times IQR_C$. This threshold helps identify correct responses that took an unusually long time, potentially indicating a less efficient problem-solving process despite arriving at the correct answer.</li>")
    html_content.append("        </ul>")
    html_content.append("        <h3 class=\"mt-4\">Labeling Logic:</h3>")
    html_content.append("        <p>The following rules are applied sequentially to assign a behavioral label:</p>")
    html_content.append("        <ol>")
    html_content.append("            <li>If `UF_C` cannot be computed (e.g., no valid correct answers for a given question/part), the label is set to <code>NA_no_correct</code> (Not Applicable - No Correct Answers).</li>")
    html_content.append("            <li>If the participant's answer is <strong>incorrect</strong>, the label is set to <code>UP</code> (Unusual Performance).</li>")
    html_content.append("            <li>If the participant's answer is <strong>correct</strong> but their `t_ij` is greater than `UF_C`, the label is also set to <code>UP</code> (Unusual Performance), indicating an unusually long time for a correct response.</li>")
    html_content.append("            <li>In all other cases (correct answer and `t_ij` ≤ `UF_C`), the label is set to <code>NP</code> (Normal Performance).</li>")
    html_content.append("        </ol>")
    html_content.append("        <h3 class=\"mt-4\">Label Distribution</h3>")
    html_content.append("        <p>The distribution of assigned behavioral labels across all valid interactions is as follows:</p>")
    html_content.append(label_counts.to_html(index=False, classes='table table-striped table-bordered'))
    if not stats_c.empty:
        html_content.append("        <h3 class=\"mt-4\">Sample of Thresholds for Correct Answers (UF_C)</h3>")
        html_content.append("        <p>This table provides a sample of the calculated Q1_C, Median_C, Q3_C, IQR_C, and UF_C values, derived exclusively from correct responses. These thresholds are used to differentiate between normal and unusual performance among correct answers.</p>")
        html_content.append(stats_c.head(20).to_html(index=False, classes='table table-striped table-bordered'))
    html_content.append("        <div class=\"section-divider\"></div>")

    html_content.append("        <h2 class=\"mt-5\">Stage 4 — Area of Interest (AOI) Features & Cognitive Phases</h2>")
    html_content.append("        <p><strong>Objective:</strong> This final processing stage extracts granular features related to specific Areas of Interest (AOIs) on the screen and delineates distinct cognitive phases (Reading and Answering) within each interaction.</p>")
    html_content.append("        <h3 class=\"mt-4\">Methodology:</h3>")
    html_content.append("        <ul>")
    html_content.append("            <li><strong>Cognitive Phase Duration Computation:</strong> The total interaction time (`t_ij`) is segmented into two primary cognitive phases: <strong>Reading Duration</strong> and <strong>Answering Duration</strong>. This segmentation is critical for understanding how participants allocate their attention during problem-solving.")
    html_content.append("                <ul>")
    html_content.append("                    <li><strong>Question→Choice Transition:</strong> The transition point from the Reading phase to the Answering phase is determined by identifying the first gaze sample that falls within any of the defined `Choice` AOIs after initially fixating on the `Question` AOI. If `BKID` (Button/Key ID) data is available, it is used to precisely mark the moment a participant interacts with an option.</li>")
    html_content.append("                    <li><strong>Midpoint Fallback:</strong> In cases where AOI transition data or `BKID` is not available or ambiguous, a fallback mechanism is employed where the midpoint of the total `t_ij` is used to approximate the transition between reading and answering phases.</li>")
    html_content.append("                </ul>")
    html_content.append("            </li>")
    html_content.append("            <li><strong>AOI Time Aggregation:</strong> For each interaction, the cumulative gaze duration within predefined Areas of Interest (AOIs) is calculated. These AOIs typically include: `Question` (the question text area), `Choice_A`, `Choice_B`, `Choice_C`, `Choice_D` (individual answer options), `Timer` (the countdown timer area), and `Submit` (the submission button area). These aggregated times provide insights into attentional distribution.</li>")
    html_content.append("        </ul>")
    cols_show = [c for c in ['participant_id','question_id','part','t_ij','label',
                             'Reading_duration_s','Answering_duration_s',
                             'Question','Choice_A','Choice_B','Choice_C','Choice_D','Timer','Submit']
                 if c in final_df.columns]
    if cols_show:
        html_content.append("        <h3 class=\"mt-4\">Sample of Final Processed Features (Stage 4)</h3>")
        html_content.append("        <p>The table below displays a sample of the enriched dataset after Stage 4, including behavioral labels, phase durations, and aggregated AOI gaze times. These features form the basis for further in-depth analysis.</p>")
        html_content.append(final_df[cols_show].head(20).to_html(index=False, classes='table table-striped table-bordered'))
    html_content.append("        <div class=\"section-divider\"></div>")

    html_content.append("        <h2 class=\"mt-5\">Key Variables and Definitions</h2>")
    html_content.append("        <p>This section provides a glossary of key variables and terms used throughout the data mining pipeline and in this report, crucial for a thorough understanding of the analysis.</p>")
    html_content.append("        <ul>")
    html_content.append("            <li><strong>`participant_id`</strong>: A unique identifier assigned to each study participant.</li>")
    html_content.append("            <li><strong>`question_id`</strong>: A unique identifier for each question presented to participants.</li>")
    html_content.append("            <li><strong>`part`</strong>: Denotes the section of the exam (e.g., 'Part 1', 'Part 2') to which a question belongs.</li>")
    html_content.append("            <li><strong>`BPOGV` (Binocular Point of Gaze Validity)</strong>: A metric indicating the validity of the recorded gaze sample. A value of 1 typically signifies valid gaze data.</li>")
    html_content.append("            <li><strong>`t_ij` (Interaction Time)</strong>: The total duration, in seconds, that participant `i` spent interacting with question `j`.</li>")
    html_content.append("            <li><strong>`Q1`, `median`, `Q3`</strong>: The first quartile, median, and third quartile of `t_ij` values, respectively, calculated per question and part.</li>")
    html_content.append("            <li><strong>`IQR` (Interquartile Range)</strong>: The difference between the third and first quartiles (`Q3 - Q1`), representing the spread of the middle 50% of `t_ij` values.</li>")
    html_content.append("            <li><strong>`LB` (Lower Bound)</strong>: A statistical threshold calculated as $Q1 - 1.5 \times IQR$, used to identify unusually short interaction times (outliers).</li>")
    html_content.append("            <li><strong>`invalid_time`</strong>: A flag indicating that an interaction's `t_ij` fell below the `LB`, suggesting an outlier.</li>")
    html_content.append("            <li><strong>`is_correct`</strong>: A binary variable (1 or 0) indicating whether the participant's answer to a question was correct.</li>")
    html_content.append("            <li><strong>`Q1_C`, `median_C`, `Q3_C`</strong>: The first quartile, median, and third quartile of `t_ij` values, calculated exclusively for <strong>correct answers</strong> per question and part.</li>")
    html_content.append("            <li><strong>`IQR_C` (Interquartile Range for Correct Answers)</strong>: The `IQR` calculated specifically for `t_ij` values of correct answers.</li>")
    html_content.append("            <li><strong>`UF_C` (Upper Fence for Correct Answers)</strong>: A statistical threshold calculated as $Q3_C + 1.5 \times IQR_C$, used to identify unusually long interaction times for correct answers.</li>")
    html_content.append("            <li><strong>`label`</strong>: The behavioral label assigned to each interaction:")
    html_content.append("                <ul>")
    html_content.append("                    <li><code>NP</code> (Normal Performance): Correct answer with `t_ij` within expected range.</li>")
    html_content.append("                    <li><code>UP</code> (Unusual Performance): Incorrect answer, or correct answer with `t_ij` exceeding `UF_C`.</li>")
    html_content.append("                    <li><code>INVALID</code>: Interaction flagged due to `invalid_time` in Stage 2.</li>")
    html_content.append("                    <li><code>NA_no_correct</code>: Not Applicable, due to insufficient correct answers to compute `UF_C`.</li>")
    html_content.append("                </ul>")
    html_content.append("            <li><strong>`Reading_duration_s`</strong>: The estimated time, in seconds, a participant spent reading the question and options.</li>")
    html_content.append("            <li><strong>`Answering_duration_s`</strong>: The estimated time, in seconds, a participant spent actively considering and selecting an answer.</li>")
    html_content.append("            <li><strong>AOI (Area of Interest)</strong>: Predefined regions on the screen (e.g., Question, Choice A, Timer, Submit) used to aggregate gaze data.</li>")
    html_content.append("        </ul>")
    html_content.append("        <div class=\"section-divider\"></div>")

    html_content.append("        <h2 class=\"mt-5\">Notes</h2>")
    html_content.append("        <ul>")
    html_content.append("            <li>If the background image is not available, the heatmap will be generated without a background.</li>")
    html_content.append("            <li>The coordinates for writing texts and rectangles on the background are read from <code>config.ini</code>; if not present, default values are used.</li>")
    html_content.append("        </ul>")
    html_content.append("        <div class=\"section-divider\"></div>")

    # Data Cleaning Diagnostics section (per professor request)
    html_content.append("        <h2 class=\"mt-5\">Data Cleaning Diagnostics</h2>")
    html_content.append("        <p>This section contains per-trial and per-participant diagnostics describing how many raw gaze samples were considered invalid and how many were retained after the cleaning rules. These reports are saved to the <code>reports/</code> folder.</p>")
    # Attempt to embed small samples of the CSV diagnostics if available
    try:
        gaze_csv = os.path.join(os.path.dirname(report_path), 'reports', 'gaze_validity_summary.csv')
        part_csv = os.path.join(os.path.dirname(report_path), 'reports', 'participant_validity_summary.csv')
        qdist_csv = os.path.join(os.path.dirname(report_path), 'reports', 'question_part_distribution.csv')
        if os.path.exists(gaze_csv):
            gv = pd.read_csv(gaze_csv)
            html_content.append("        <h3 class=\"mt-3\">Per-Trial Gaze Validity (sample)</h3>")
            html_content.append(gv.head(20).to_html(index=False, classes='table table-striped table-bordered'))
        else:
            html_content.append("        <p><em>gaze_validity_summary.csv not found.</em></p>")
        if os.path.exists(part_csv):
            pv = pd.read_csv(part_csv)
            html_content.append("        <h3 class=\"mt-3\">Per-Participant Validity Summary</h3>")
            html_content.append(pv.to_html(index=False, classes='table table-striped table-bordered'))
        else:
            html_content.append("        <p><em>participant_validity_summary.csv not found.</em></p>")
        if os.path.exists(qdist_csv):
            qd = pd.read_csv(qdist_csv)
            html_content.append("        <h3 class=\"mt-3\">Question-Part Distribution</h3>")
            html_content.append(qd.to_html(index=False, classes='table table-striped table-bordered'))
        else:
            html_content.append("        <p><em>question_part_distribution.csv not found.</em></p>")
    except Exception:
        html_content.append("        <p>Could not load diagnostic CSVs for embedding.</p>")

    html_content.append("        <div class=\"section-divider\"></div>")

    html_content.append("        <h2 class=\"mt-5\">Overlay Coordinates (from config.ini)</h2>")
    if config is None:
        html_content.append("        <p>No configuration object provided; default overlay coordinates were used.</p>")
    else:
        overlay_section = config['Overlay'] if 'Overlay' in config else {}
        if overlay_section:
            overlay_df = pd.DataFrame(overlay_section.items(), columns=['Key', 'Value'])
            html_content.append(overlay_df.to_html(index=False, classes='table table-striped table-bordered'))
        else:
            html_content.append("        <p>No [Overlay] section found in config.ini — defaults used.</p>")
    html_content.append("        <div class=\"section-divider\"></div>")

    html_content.append("        <h2 class=\"mt-5\">Per-stage Analysis Summary</h2>")
    html_content.append("        <p>This section provides a concise summary of key findings and statistics derived from each stage of the data processing pipeline.</p>")

    html_content.append("        <h3 class=\"mt-4\">Stage 1 — Data Cleaning & t_ij Computation Summary</h3>")
    html_content.append(f"        <p><strong>Total Valid Interactions (post-cleaning):</strong> {int(labeled_df.shape[0] if 'label' in labeled_df.columns else 0)} records.</p>")
    if 't_ij' in final_df.columns:
        html_content.append(f"        <p><strong>Interaction Time (t_ij) Statistics:</strong></p>")
        html_content.append("        <ul>")
        html_content.append(f"            <li>Mean t_ij: {final_df['t_ij'].mean():.2f} seconds</li>")
        html_content.append(f"            <li>Median t_ij: {final_df['t_ij'].median():.2f} seconds</li>")
        html_content.append(f"            <li>Standard Deviation of t_ij: {final_df['t_ij'].std():.2f} seconds</li>")
        html_content.append("        </ul>")
        html_content.append("        <p>These statistics indicate the central tendency and variability of participant engagement times after initial data cleaning and filtering of very short interactions.</p>")
    html_content.append("        <div class=\"section-divider\"></div>")

    html_content.append("        <h3 class=\"mt-4\">Stage 2 — Outlier Detection Summary</h3>")
    if not stats_all.empty:
        html_content.append(f"        <p><strong>Number of Question-Part Groups with Computed Lower Bounds (LB):</strong> {stats_all.shape[0]}.</p>")
        html_content.append("        <p>The outlier detection process identified interactions with unusually short durations, which are critical for understanding potentially disengaged or rushed responses. A sample of the computed LB thresholds is provided above.</p>")
    else:
        html_content.append("        <p>No Lower Bound (LB) thresholds were computed, possibly due to insufficient data for statistical analysis in this stage.</p>")
    html_content.append("        <div class=\"section-divider\"></div>")

    html_content.append("        <h3 class=\"mt-4\">Stage 3 — Behavioral Labeling Summary</h3>")
    if 'label' in labeled_df.columns:
        label_counts_dict = labeled_df['label'].value_counts().to_dict()
        html_content.append("        <p><strong>Distribution of Behavioral Labels:</strong></p>")
        html_content.append("        <ul>")
        for k, v in label_counts_dict.items():
            html_content.append(f"            <li><strong>{k}</strong>: {v} instances.</li>")
        html_content.append("        </ul>")
        html_content.append("        <p>This distribution provides a high-level overview of participant performance and engagement patterns, categorizing responses into Normal Performance (NP), Unusual Performance (UP), Invalid interactions, and cases where correct answer thresholds could not be established (NA_no_correct).</p>")
    else:
        html_content.append("        <p>No behavioral labels were generated, indicating a potential issue in the labeling stage or insufficient valid data.</p>")
    html_content.append("        <div class=\"section-divider\"></div>")

    html_content.append("        <h3 class=\"mt-4\">Stage 4 — AOI Features & Cognitive Phases Summary</h3>")
    if 'Reading_duration_s' in final_df.columns or 'Answering_duration_s' in final_df.columns:
        rd = final_df['Reading_duration_s'].dropna() if 'Reading_duration_s' in final_df.columns else pd.Series(dtype=float)
        ad = final_df['Answering_duration_s'].dropna() if 'Answering_duration_s' in final_df.columns else pd.Series(dtype=float)
        html_content.append("        <p><strong>Cognitive Phase Durations:</strong></p>")
        html_content.append("        <ul>")
        if not rd.empty:
            html_content.append(f"            <li><strong>Reading Duration (s):</strong> Mean = {rd.mean():.2f}, Median = {rd.median():.2f}, N = {len(rd)}</li>")
        if not ad.empty:
            html_content.append(f"            <li><strong>Answering Duration (s):</strong> Mean = {ad.mean():.2f}, Median = {ad.median():.2f}, N = {len(ad)}</li>")
        html_content.append("        </ul>")
        html_content.append("        <p>These metrics offer insights into how participants divide their attention between understanding the question and formulating a response. The aggregated AOI times (presented in the sample table above) further detail specific attentional foci.</p>")
    else:
        html_content.append("        <p>No cognitive phase duration features were found in the final output, suggesting an issue in the feature engineering stage.</p>")
    html_content.append("        <div class=\"section-divider\"></div>")

    html_content.append("        <h2 class=\"mt-5\">Overall Statistical Summary</h2>")
    html_content.append("        <p>This section provides a high-level statistical overview of the entire dataset and the results of the pipeline.</p>")
    html_content.append("        <ul>")
    html_content.append(f"            <li><strong>Total Participants Analyzed:</strong> {n_participants}</li>")
    html_content.append(f"            <li><strong>Total Questions Analyzed:</strong> {n_questions}</li>")
    if 't_ij' in final_df.columns:
        html_content.append(f"            <li><strong>Overall Mean Interaction Time (t_ij):</strong> {final_df['t_ij'].mean():.2f} seconds</li>")
        html_content.append(f"            <li><strong>Overall Median Interaction Time (t_ij):</strong> {final_df['t_ij'].median():.2f} seconds</li>")
    if 'label' in labeled_df.columns:
        html_content.append("            <li><strong>Overall Behavioral Label Distribution:</strong></li>")
        html_content.append("            <ul>")
        for k, v in label_counts.set_index('label')['count'].to_dict().items():
            html_content.append(f"                <li>{k}: {v} instances</li>")
        html_content.append("            </ul>")
    html_content.append("        </ul>")
    html_content.append("        <div class=\"section-divider\"></div>")

    # Add all generated images to the report
    html_content.append("        <h2 class=\"mt-5\">Visualizations</h2>")

    # AOI Summary per Question (New Plot)
    aoi_summary_path = os.path.join(viz_dir, 'aoi_summary_per_question.png')
    if os.path.exists(aoi_summary_path):
        aoi_summary_rel_path = os.path.relpath(aoi_summary_path, os.path.dirname(report_path)).replace('\\', '/')
        html_content.append("        <h3 class=\"mt-4\">AOI Time Summary per Question</h3>")
        html_content.append(f"        <img src=\"{aoi_summary_rel_path}\" alt=\"AOI Summary per Question\" class=\"img-fluid\" onclick=\"openModal(this)\">")
        # If a CSV summary exists, include it as an HTML table below the image
        csv_summary_path = os.path.join(viz_dir, 'avg_aoi_per_question.csv')
        if os.path.exists(csv_summary_path):
            try:
                # Read the full CSV to embed in the report
                summary_table_df = pd.read_csv(csv_summary_path)
                html_content.append("        <h4 class=\"mt-3\">Numeric Summary (per question)</h4>")
                html_content.append(summary_table_df.to_html(index=False, classes='table table-striped table-bordered'))
            except Exception as e:
                print(f"Could not read or embed AOI summary CSV: {e}")
    
    # Summary Plots
    if summary_img_path and os.path.exists(summary_img_path):
        summary_img_rel_path = os.path.relpath(summary_img_path, os.path.dirname(report_path)).replace('\\', '/')
        html_content.append("        <h3 class=\"mt-4\">Pipeline Summary Plots</h3>")
        html_content.append(f"        <img src=\"{summary_img_rel_path}\" alt=\"Pipeline Summary Plots\" class=\"img-fluid\">")

    # Heatmaps and Scatterplots side-by-side
    heatmaps_base_dir = os.path.join(viz_dir, 'heatmaps')
    scatterplots_base_dir = os.path.join(viz_dir, 'points')

    all_participant_folders = sorted(list(set(os.listdir(heatmaps_base_dir) if os.path.exists(heatmaps_base_dir) else []) |\
                                         set(os.listdir(scatterplots_base_dir) if os.path.exists(scatterplots_base_dir) else [])))

    # AOI Time per Question
    aoi_q_path = os.path.join(viz_dir, 'aoi_time_per_question.png')
    if os.path.exists(aoi_q_path):
        aoi_q_rel_path = os.path.relpath(aoi_q_path, os.path.dirname(report_path)).replace('\\', '/')
        html_content.append("        <h3 class=\"mt-4\">AOI Time per Question</h3>")
        html_content.append(f"        <img src=\"{aoi_q_rel_path}\" alt=\"AOI Time per Question\" class=\"img-fluid\">")

    # AOI Time per Label
    aoi_l_path = os.path.join(viz_dir, 'aoi_time_per_label.png')
    if os.path.exists(aoi_l_path):
        aoi_l_rel_path = os.path.relpath(aoi_l_path, os.path.dirname(report_path)).replace('\\', '/')
        html_content.append("        <h3 class=\"mt-4\">AOI Time per Label</h3>")
        html_content.append(f"        <img src=\"{aoi_l_rel_path}\" alt=\"{aoi_l_path}\" class=\"img-fluid\">")

    if all_participant_folders:
        html_content.append("        <h3 class=\"mt-4\">Gaze Heatmaps and Scatterplots</h3>")
        for p_folder in all_participant_folders:
            html_content.append(f"        <h4>{p_folder.replace('participant_', 'Participant ')}</h4>")
            
            participant_heatmaps_dir = os.path.join(heatmaps_base_dir, p_folder)
            participant_scatterplots_dir = os.path.join(scatterplots_base_dir, p_folder)

            all_part_folders = sorted(list(set(os.listdir(participant_heatmaps_dir) if os.path.exists(participant_heatmaps_dir) else []) |\
                                          set(os.listdir(participant_scatterplots_dir) if os.path.exists(participant_scatterplots_dir) else [])))
            
            for part_folder in all_part_folders:
                html_content.append(f"        <h5>{part_folder.replace('_', ' ')}</h5>")
                
                part_heatmaps_path = os.path.join(participant_heatmaps_dir, part_folder)
                part_scatterplots_path = os.path.join(participant_scatterplots_dir, part_folder)

                heatmap_files = sorted([f for f in os.listdir(part_heatmaps_path) if f.endswith('.png')] if os.path.exists(part_heatmaps_path) else [])
                scatterplot_files = sorted([f for f in os.listdir(part_scatterplots_path) if f.endswith('.png')] if os.path.exists(part_scatterplots_path) else [])

                all_q_ids = sorted(list(set([f.replace('gaze_heatmap_', '').replace('.png', '') for f in heatmap_files]) |\
                                         set([f.replace('gaze_point_', '').replace('.png', '') for f in scatterplot_files])))

                for q_id in all_q_ids:
                    heatmap_img_file = f'gaze_heatmap_{q_id}.png'
                    scatterplot_img_file = f'gaze_point_{q_id}.png'

                    heatmap_exists = os.path.exists(os.path.join(part_heatmaps_path, heatmap_img_file))
                    scatterplot_exists = os.path.exists(os.path.join(part_scatterplots_path, scatterplot_img_file))

                    html_content.append(f"        <h6>Question {q_id}</h6>")
                    html_content.append("        <div style=\"display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-bottom: 20px;\">")

                    if heatmap_exists:
                        heatmap_img_path = os.path.join(part_heatmaps_path, heatmap_img_file)
                        heatmap_rel_path = os.path.relpath(heatmap_img_path, os.path.dirname(report_path)).replace('\\', '/')
                        html_content.append(f"            <div style=\"flex: 1; min-width: 45%;\"><img src=\"{heatmap_rel_path}\" alt=\"Heatmap {q_id}\" class=\"img-fluid\"></div>")
                    
                    if scatterplot_exists:
                        scatterplot_img_path = os.path.join(part_scatterplots_path, scatterplot_img_file)
                        scatterplot_rel_path = os.path.relpath(scatterplot_img_path, os.path.dirname(report_path)).replace('\\', '/')
                        html_content.append(f"            <div style=\"flex: 1; min-width: 45%;\"><img src=\"{scatterplot_rel_path}\" alt=\"Scatterplot {q_id}\" class=\"img-fluid\"></div>")
                    
                    html_content.append("        </div>")

    html_content.append("</div>")
    html_content.append("</body>")
    html_content.append("<div id=\"myModal\" class=\"modal\">")
    html_content.append("        <span class=\"close\">&times;</span>")
    html_content.append("        <img class=\"modal-content\" id=\"img01\">")
    html_content.append("        <div id=\"caption\"></div>")
    html_content.append("    </div>")
    html_content.append("<script>")
    html_content.append("        // Get the modal")
    html_content.append("        var modal = document.getElementById(\"myModal\");")
    html_content.append("")
    html_content.append("        // Get the image and insert it inside the modal - use its \"alt\" text as a caption")
    html_content.append("        var modalImg = document.getElementById(\"img01\");")
    html_content.append("        var captionText = document.getElementById(\"caption\");")
    html_content.append("")
    html_content.append("        document.querySelectorAll('img').forEach(item => {")
    html_content.append("            item.onclick = function(){")
    html_content.append("                modal.style.display = \"block\";")
    html_content.append("                modalImg.src = this.src;")
    html_content.append("                captionText.innerHTML = this.alt;")
    html_content.append("            }")
    html_content.append("        });")
    html_content.append("")
    html_content.append("        // Get the <span> element that closes the modal")
    html_content.append("        var span = document.getElementsByClassName(\"close\")[0];")
    html_content.append("")
    html_content.append("        // When the user clicks on <span> (x), close the modal")
    html_content.append("        span.onclick = function() {")
    html_content.append("            modal.style.display = \"none\";")
    html_content.append("        }")
    html_content.append("")
    html_content.append("        // Close the modal when clicking outside the image")
    html_content.append("        modal.onclick = function(event) {")
    html_content.append("            if (event.target == modal) {")
    html_content.append("                modal.style.display = \"none\";")
    html_content.append("            }")
    html_content.append("        }")
    html_content.append("    </script>")
    html_content.append("</html>")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_content))

# ======================================================================================
# >>> ADDED: PROGRESS MONITORING UI
# ======================================================================================

class ProgressMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Pipeline Progress")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.pipeline_thread = None  # Will be set externally

        self.progress_var = tk.DoubleVar()
        self.stage_progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar()
        self.stage_status_var = tk.StringVar()
        self.time_var = tk.StringVar()
        self.progress_percent_var = tk.StringVar()
        self.stage_progress_percent_var = tk.StringVar()
        self.cancel_event = threading.Event()

        self.status_var.set("Initializing...")
        self.stage_status_var.set("")
        self.time_var.set("Elapsed Time: 00:00:00")
        self.progress_percent_var.set("0%")
        self.stage_progress_percent_var.set("0%")

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Processing Status:").pack(anchor='w', pady=(0, 5))
        ttk.Label(main_frame, textvariable=self.status_var, wraplength=560).pack(anchor='w', fill=tk.X, pady=(0, 6))

        # Overall progress
        ttk.Label(main_frame, text="Overall Progress:").pack(anchor='w')
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=6)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(progress_frame, textvariable=self.progress_percent_var).pack(side=tk.RIGHT, padx=(5, 0))

        # Stage progress
        ttk.Label(main_frame, text="Current Stage:").pack(anchor='w')
        ttk.Label(main_frame, textvariable=self.stage_status_var, wraplength=560).pack(anchor='w', fill=tk.X, pady=(0, 4))
        stage_progress_frame = ttk.Frame(main_frame)
        stage_progress_frame.pack(fill=tk.X, pady=6)
        self.stage_progress_bar = ttk.Progressbar(stage_progress_frame, variable=self.stage_progress_var, maximum=100)
        self.stage_progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(stage_progress_frame, textvariable=self.stage_progress_percent_var).pack(side=tk.RIGHT, padx=(5, 0))

        # Details log area
        ttk.Label(main_frame, text="Details:").pack(anchor='w', pady=(6, 0))
        details_frame = ttk.Frame(main_frame)
        details_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 6))

        # Create the Text widget with both scrollbars INSIDE the same frame
        self.details_text = tk.Text(details_frame, height=6, wrap='word', state='normal', exportselection=True)
        self.details_text.grid(row=0, column=0, sticky="nsew")

        # Vertical scrollbar
        v_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.details_text.config(yscrollcommand=v_scrollbar.set)

        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(details_frame, orient=tk.HORIZONTAL, command=self.details_text.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.details_text.config(xscrollcommand=h_scrollbar.set)

        # Configure grid weights so the Text widget expands
        details_frame.rowconfigure(0, weight=1)
        details_frame.columnconfigure(0, weight=1)

        ttk.Label(main_frame, textvariable=self.time_var).pack(anchor='e', pady=(2, 6))

        self.cancel_button = ttk.Button(main_frame, text="Cancel", command=self.cancel)
        self.cancel_button.pack(side=tk.LEFT, padx=5, pady=6)

        self.open_report_button = ttk.Button(main_frame, text="Open Report", command=self._open_report, state=tk.DISABLED)
        self.open_report_button.pack(side=tk.RIGHT, padx=5, pady=6)

        self.report_path = None

        self.start_time = None

        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        # اگر مقدار صفر بود، از مقدار پیشنهادی استفاده کن
        if width == 0 or height == 0:
            width = self.root.winfo_reqwidth()
            height = self.root.winfo_reqheight()
        x = (self.root.winfo_screenwidth() - width) // 2
        y = (self.root.winfo_screenheight() - height) // 2
        self.root.geometry(f"+{x}+{y}")

    def _open_report(self):
        import webbrowser
        if self.report_path and os.path.exists(self.report_path):
            webbrowser.open(self.report_path)
        else:
            import tkinter.messagebox as messagebox
            messagebox.showwarning("File Not Found", "The report file could not be found.")

    def _on_closing(self):
        # Allow closing the window if the pipeline is not running or has been cancelled
        if self.pipeline_thread is None or not self.pipeline_thread.is_alive() or self.cancel_event.is_set():
            self.stop_timer()
            self.root.quit()
            self.root.destroy()
        else:
            # Optionally, show a warning if the pipeline is still running
            # messagebox.showwarning("Pipeline Running", "Cannot close while pipeline is active. Please cancel first.")
            pass # Prevent closing the window with the 'X' button while running
            pass # Prevent closing the window with the 'X' button while running

    def start_timer(self):
        self.start_time = time.time()
        self._update_timer()

    def stop_timer(self):
        # stops the periodic updates
        self.start_time = None

    def _update_timer(self):
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        self.time_var.set(f"Elapsed Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        # Schedule next update
        try:
            self.root.after(200, self._update_timer)
        except Exception:
            # GUI may have been destroyed
            pass

    def update_progress(self, value, status_text):
        """Update overall progress and main status text."""
        try:
            self.progress_var.set(value)
            self.progress_percent_var.set(f"{int(value)}%")
            self.status_var.set(status_text)
        except Exception:
            pass

    def update_stage_progress(self, value, stage_text=None):
        """Update stage-level progress bar and its text."""
        try:
            self.stage_progress_var.set(value)
            self.stage_progress_percent_var.set(f"{int(value)}%")
            if stage_text is not None:
                self.stage_status_var.set(stage_text)
        except Exception:
            pass

    def append_log(self, text):
        try:
            self.details_text.insert(tk.END, f"{text}\n")
            self.details_text.see(tk.END)
            # حالت را به 'normal' نگه می‌داریم تا قابل انتخاب و کپی باشد
        except Exception:
            pass

    def cancel(self):
        self.cancel_event.set()
        self.status_var.set("Cancellation requested...")
        self.append_log("Cancellation requested by user.")
        self.cancel_button.config(state=tk.DISABLED)
        self.stop_timer() # Stop the timer when cancel is pressed
        # end program execution in the main loop
        self.root.quit()
        self.root.destroy()

    def is_cancelled(self):
        return self.cancel_event.is_set()

    def close(self):
        # Stop the timer and close the window
        self.stop_timer()
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

def run_pipeline_in_thread(pipeline_func, progress_queue):
    try:
        pipeline_func(progress_queue)
    except Exception as e:
        progress_queue.put(("error", f"An error occurred: {e}"))

# =====================
# PIPELINE PROGRESS MANAGEMENT (Professional)
# =====================
PIPELINE_STAGES = [
    ("Initialization", ["Create output directories", "Load backgrounds"]),
    ("Data Loading", ["Load config", "Load participants", "Load questions/answers", "Filter data"]),
    ("Data Cleaning", ["Remove invalid gaze", "Compute t_ij"]),
    ("Outlier Detection", ["Compute stats", "Flag outliers", "Save stage2 data"]),
    ("Labeling", ["Apply behavioral labels", "Save stage3 data"]),
    ("Feature Engineering", ["AOI assignment", "Phase extraction", "Save stage4 data"]),
    ("Visualization", ["Summary plots", "AOI Summary", "Heatmaps", "Scatter plots", "AOI per question", "AOI per label"]),
    ("Report", ["Generate markdown report"])
]

def get_total_steps():
    return sum(len(subs) for _, subs in PIPELINE_STAGES)

def update_pipeline_progress(progress_queue, stage_idx, substage_idx, status_text):
    total_steps = get_total_steps()
    current_step = sum(len(subs) for _, subs in PIPELINE_STAGES[:stage_idx]) + substage_idx + 1
    overall_percent = int(100 * current_step / total_steps)
    stage_name, substages = PIPELINE_STAGES[stage_idx]
    stage_percent = int(100 * (substage_idx + 1) / len(substages))
    progress_queue.put(("progress", (overall_percent, f"[{stage_name}] {status_text}")))
    progress_queue.put(("stage_progress", (stage_percent, status_text)))

# ======================================================================================
# MAIN ENTRY POINT
# ======================================================================================

print("--- Checking main entry point ---")

def main():
    """Main function to run the entire data mining pipeline."""
    
    participant_range, question_range, selected_plots, save_stage_outputs = get_input_range()
    if not participant_range or not question_range:
        print("Pipeline cancelled by user or invalid range provided.")
        return

    # --- Progress UI Setup ---
    global monitor
    progress_root = tk.Tk()
    monitor = ProgressMonitor(progress_root)
    progress_queue = queue.Queue()
    cancel_event = monitor.cancel_event  # Get the event from the monitor

    # Redirect stdout to the logger
    original_stdout = sys.stdout
    sys.stdout = Logger(progress_queue, original_stdout)

    def process_queue():
        try:
            # Drain the queue on each call
            while True:
                message_type, value = progress_queue.get_nowait()
                if message_type == "progress":
                    monitor.update_progress(value[0], value[1])
                    monitor.append_log(value[1])  # Add this line
                elif message_type == "stage_progress":
                    monitor.update_stage_progress(value[0], value[1])
                    monitor.append_log(value[1])  # Add this line
                elif message_type == "log":
                    monitor.append_log(value)
                elif message_type == "error":
                    monitor.update_progress(100, f"Error: {value}")
                    monitor.append_log(f"Error: {value}")
                    monitor.stop_timer()
                    monitor.cancel_button.config(text="Close", command=monitor.close)
                    monitor.cancel_button.config(state=tk.NORMAL)
                    return  # Stop processing queue
                elif message_type == "done":
                    monitor.update_progress(100, "Pipeline finished successfully!")
                    monitor.append_log("Pipeline finished successfully!")
                    monitor.stop_timer()
                    # Enable Open Report button
                    monitor.open_report_button.config(state=tk.NORMAL)
                    monitor.cancel_button.config(text="Close", command=monitor.close)
                    monitor.cancel_button.config(state=tk.NORMAL)
                    return  # Stop processing queue
        except queue.Empty:
            pass # No more messages in the queue for now
        
        # If the thread is still running, schedule the next check
        if pipeline_thread.is_alive():
            progress_root.after(100, process_queue)
        else:
            # The thread has finished. Check if it finished unexpectedly.
            # A normal finish would have sent a "done" or "error" message, which would have stopped the loop via `return`.
            if not monitor.is_cancelled() and monitor.progress_var.get() < 100:
                monitor.update_progress(100, "Process finished or terminated unexpectedly.")
                monitor.append_log("Process finished or terminated unexpectedly.")
                monitor.stop_timer()
                monitor.cancel_button.config(text="Close", command=monitor.close)
                monitor.cancel_button.config(state=tk.NORMAL)
            
    # --- Pipeline Logic (to be run in a separate thread) ---
    def pipeline_logic(progress_queue, cancel_event):
        summary_img_path = None
        try:
            # Stage 0: Initialization
            update_pipeline_progress(progress_queue, 0, 0, "Creating output directories...")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            viz_dir = os.path.join(script_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            intermediate_output_dir = os.path.join(script_dir, 'intermediate_processed_data')
            os.makedirs(intermediate_output_dir, exist_ok=True)
            reports_dir = os.path.join(script_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            update_pipeline_progress(progress_queue, 0, 1, "Loading background images...")
            bg_image_part1 = None
            bg_image_part2 = None
            if selected_plots.get('heatmaps') or selected_plots.get('scatterplots'):
                # First check if the background images exist
                bg_path1 = os.path.join(script_dir, 'part1_background.png')
                bg_path2 = os.path.join(script_dir, 'part2_background.png')
                
                bg_missing = []
                if not os.path.exists(bg_path1):
                    bg_missing.append("Part 1")
                if not os.path.exists(bg_path2):
                    bg_missing.append("Part 2")
                
                if bg_missing:
                    print(f"\nWarning: Background images missing for {' and '.join(bg_missing)}:")
                    print(f"Expected files:")
                    if "Part 1" in bg_missing:
                        print(f"- {bg_path1}")
                    if "Part 2" in bg_missing:
                        print(f"- {bg_path2}")
                    print("Please ensure these files exist to see backgrounds in visualizations.\n")
                
                try:
                    if os.path.exists(bg_path1):
                        bg_image_part1 = plt.imread(bg_path1)
                        print("Successfully loaded Part 1 background image")
                except Exception as e:
                    print(f"Error loading Part 1 background: {str(e)}")

                try:
                    if os.path.exists(bg_path2):
                        bg_image_part2 = plt.imread(bg_path2)
                        print("Successfully loaded Part 2 background image")
                except Exception as e:
                    print(f"Error loading Part 2 background: {str(e)}")
            # Stage 1: Data Loading
            update_pipeline_progress(progress_queue, 1, 0, "Loading config...")
            config = load_config(os.path.join(script_dir, 'config.ini'))
            output_dir = os.path.join(script_dir, config.get('Paths', 'output_dir', fallback='outputs'))
            questions_path = os.path.join(script_dir, config.get('Paths', 'questions_json', fallback='questions.json'))
            question_exams_dir = os.path.join(script_dir, config.get('Paths', 'question_exams_dir', fallback='question_exams'))
            update_pipeline_progress(progress_queue, 1, 1, "Loading participant data...")
            raw_df = load_all_participant_data(output_dir)
            if raw_df.empty:
                progress_queue.put(("error", "No participant data loaded."))
                return
            update_pipeline_progress(progress_queue, 1, 2, "Loading question and answer data...")
            answer_data, correct_answers = load_question_data(questions_path, output_dir)
            part_data = load_question_parts(question_exams_dir, output_dir)
            question_texts = load_question_texts(questions_path)
            update_pipeline_progress(progress_queue, 1, 3, "Filtering data based on user input...")
            participant_ids = [f"participant_{i}" for i in participant_range]
            question_ids = [f"Q{i}" for i in question_range]
            raw_df = raw_df[raw_df['participant_id'].isin(participant_ids) & raw_df['question_id'].isin(question_ids)].copy()
            
            # Write debug info to a file
            debug_file_path = os.path.join(reports_dir, 'debug_participants.txt')
            with open(debug_file_path, 'w') as f:
                f.write(f"Unique participants in raw_df after initial filtering: {raw_df['participant_id'].unique()}\n")

            if raw_df.empty:
                progress_queue.put(("error", "No data after filtering participants/questions."))
                return
            raw_df = pd.merge(raw_df, part_data, on=['participant_id', 'question_id'], how='left')
            answer_data = pd.merge(answer_data, part_data, on=['participant_id', 'question_id'], how='left')
            if cancel_event.is_set():
                return
            # Retrieve time_cap_s and exclude_censored from config
            time_cap_s_str = config.get('Analysis', 'time_cap_s', fallback='')
            time_cap_s = None
            if time_cap_s_str:
                try:
                    time_cap_s = float(time_cap_s_str)
                except ValueError:
                    print(f"Warning: Invalid time_cap_s '{time_cap_s_str}' in config.ini. Using no time cap.")
                    time_cap_s = None
            
            exclude_censored = config.getboolean('Analysis', 'exclude_censored', fallback=False)

            # Get invalid gaze threshold from config, with a default
            try:
                invalid_gaze_threshold = config.getfloat('Analysis', 'invalid_gaze_threshold')
            except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
                invalid_gaze_threshold = 0.3  # Default threshold
                print(f"Warning: 'invalid_gaze_threshold' not found or invalid in config.ini. Using default: {invalid_gaze_threshold}")

            # Read consecutive zero threshold and LB multiplier from config
            try:
                consecutive_zero_threshold = int(config.getint('Analysis', 'consecutive_zero_threshold', fallback=5))
            except Exception:
                consecutive_zero_threshold = 5

            try:
                lb_multiplier = float(config.get('Analysis', 'lb_multiplier', fallback='1.5'))
            except Exception:
                lb_multiplier = 1.5

            # Stage 2: Data Cleaning
            update_pipeline_progress(progress_queue, 2, 0, "Removing invalid gaze samples...")
            time_df, gaze_validity_stats, participant_summary = clean_and_prepare_data(raw_df.copy(), invalid_gaze_threshold=float(invalid_gaze_threshold), consecutive_zero_threshold=consecutive_zero_threshold, report_dir=reports_dir)
            # Save diagnostics already written inside function; also save question distribution
            qdist = question_part_distribution(question_exams_dir, output_dir)
            try:
                qdist.to_csv(os.path.join(reports_dir, 'question_part_distribution.csv'), index=False)
                print(f"Saved question part distribution to {os.path.join(reports_dir, 'question_part_distribution.csv')}")
            except Exception as e:
                print(f"Could not save question part distribution: {e}")
            update_pipeline_progress(progress_queue, 2, 1, "Computing t_ij per question...")
            if save_stage_outputs:
                time_df.to_csv(os.path.join(intermediate_output_dir, 'stage1_time_data.csv'), index=False)
            if cancel_event.is_set():
                return
            # Stage 3: Outlier Detection
            update_pipeline_progress(progress_queue, 3, 0, "Computing outlier stats...")
            outlier_df, stats_all = detect_outliers(time_df.copy(), time_cap_s=time_cap_s, exclude_censored=exclude_censored, iqr_multiplier=lb_multiplier)
            update_pipeline_progress(progress_queue, 3, 1, "Flagging outliers...")
            if save_stage_outputs:
                outlier_df.to_csv(os.path.join(intermediate_output_dir, 'stage2_outlier_data.csv'), index=False)
                stats_all.to_csv(os.path.join(intermediate_output_dir, 'stage2_outlier_stats.csv'), index=False)
            if cancel_event.is_set():
                return
            # Stage 4: Labeling
            update_pipeline_progress(progress_queue, 4, 0, "Applying behavioral labels...")
            labeled_df, stats_c = apply_behavioral_labels(outlier_df.copy(), answer_data.copy(), time_cap_s=time_cap_s, exclude_censored=exclude_censored, iqr_multiplier=lb_multiplier)
            update_pipeline_progress(progress_queue, 4, 1, "Saving Stage 3 data...")
            if save_stage_outputs:
                labeled_df.to_csv(os.path.join(intermediate_output_dir, 'stage3_labeled_data.csv'), index=False)
                stats_c.to_csv(os.path.join(intermediate_output_dir, 'stage3_correct_stats.csv'), index=False)
            if cancel_event.is_set():
                return
            # Stage 5: Feature Engineering
            update_pipeline_progress(progress_queue, 5, 0, "Assigning AOIs and extracting phases...")
            processed_df = engineer_features(raw_df.copy(), labeled_df.copy())
            update_pipeline_progress(progress_queue, 5, 1, "Saving Stage 4 data...")
            if save_stage_outputs:
                processed_df.to_csv(os.path.join(intermediate_output_dir, 'stage4_processed_data.csv'), index=False)
            # ذخیره داده نهایی در processed_data/final_processed_data.csv
            try:
                os.makedirs(os.path.join(script_dir, 'processed_data'), exist_ok=True)
                final_csv_path = os.path.join(script_dir, 'processed_data', 'final_processed_data.csv')
                processed_df.to_csv(final_csv_path, index=False, encoding='utf-8')
                progress_queue.put(("log", f"Final processed data saved to {final_csv_path}"))
            except Exception as e:
                progress_queue.put(("log", f"Error saving final processed data: {e}"))

            # نمایش نمودار نهایی (توزیع t_ij یا برچسب‌ها)
            try:
                plt.figure(figsize=(10, 6))
                if 'label' in processed_df.columns:
                    sns.countplot(data=processed_df, x='label', order=['NP', 'UP', 'INVALID', 'NA_no_correct'])
                    plt.title('Distribution of Behavioral Labels (Final Data)')
                    plt.xlabel('Label')
                    plt.ylabel('Count')
                elif 't_ij' in processed_df.columns:
                    sns.boxplot(data=processed_df, x='part', y='t_ij')
                    plt.title('Distribution of t_ij per Part (Final Data)')
                    plt.xlabel('Exam Part')
                    plt.ylabel('t_ij (seconds)')
                else:
                    plt.text(0.5, 0.5, 'No suitable data for final plot', ha='center', va='center')
                plt.tight_layout()
                plt.show()
                progress_queue.put(("log", "Final summary plot displayed."))
            except Exception as e:
                progress_queue.put(("log", f"Error displaying final summary plot: {e}"))
            if cancel_event.is_set():
                return
            # Stage 6: Visualization
            vis_idx = 0
            if selected_plots.get('summary_plots'):
                update_pipeline_progress(progress_queue, 6, vis_idx, "Generating summary plots...")
                fig_summary, axes_summary = plt.subplots(2, 2, figsize=(20, 15))
                fig_summary.suptitle('Pipeline Stage Summaries', fontsize=20)
                visualize_stage1(time_df, axes_summary[0, 0])
                visualize_stage2(outlier_df, axes_summary[0, 1])
                visualize_stage3(labeled_df, axes_summary[1, 0])
                visualize_stage4(processed_df, axes_summary[1, 1])
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                summary_img_path = os.path.join(viz_dir, 'pipeline_summary.png')
                plt.savefig(summary_img_path)
                plt.close(fig_summary)
                vis_idx += 1
            # AOI summary per question (new plot) - ensure it's generated before other per-question visuals
            if selected_plots.get('aoi_summary_per_question'):
                update_pipeline_progress(progress_queue, 6, vis_idx, "Generating AOI summary per question...")
                try:
                    visualize_aoi_summary_per_question(processed_df, correct_answers, viz_dir, progress_queue, cancel_event)
                except Exception as e:
                    progress_queue.put(("log", f"Error generating AOI summary per question: {e}"))
                vis_idx += 1
            if selected_plots.get('heatmaps'):
                update_pipeline_progress(progress_queue, 6, vis_idx, "Generating heatmaps...")
                visualize_heatmaps(raw_df, viz_dir, question_texts, bg_image_part1, bg_image_part2, config, progress_queue, cancel_event)
                vis_idx += 1
            if selected_plots.get('scatterplots'):
                update_pipeline_progress(progress_queue, 6, vis_idx, "Generating scatter plots...")
                visualize_scatterplots(raw_df, viz_dir, question_texts, bg_image_part1, bg_image_part2, config, progress_queue, cancel_event)
                vis_idx += 1
            if selected_plots.get('aoi_per_question'):
                update_pipeline_progress(progress_queue, 6, vis_idx, "Generating AOI time per question chart...")
                visualize_aoi_time_per_question(processed_df, viz_dir, question_texts, progress_queue, cancel_event)
                vis_idx += 1
            if selected_plots.get('aoi_per_label'):
                update_pipeline_progress(progress_queue, 6, vis_idx, "Generating AOI time per label chart...")
                visualize_aoi_time_per_label(processed_df, viz_dir, progress_queue, cancel_event)
                vis_idx += 1
            # Stage 7: Report
            update_pipeline_progress(progress_queue, 7, 0, "Generating HTML report...")
            
            # Calculate final statistics
            n_participants = processed_df['participant_id'].nunique()
            n_questions = processed_df['question_id'].nunique()
            label_counts = labeled_df['label'].value_counts(dropna=False).rename_axis('label').reset_index(name='count')

            # Add overall statistical summary to the progress monitor
            summary_log = "\n--- Overall Statistical Summary ---\n"
            summary_log += f"Total Participants Analyzed: {n_participants}\n"
            summary_log += f"Total Questions Analyzed: {n_questions}\n"
            if 't_ij' in processed_df.columns:
                summary_log += f"Overall Mean Interaction Time (t_ij): {processed_df['t_ij'].mean():.2f} seconds\n"
                summary_log += f"Overall Median Interaction Time (t_ij): {processed_df['t_ij'].median():.2f} seconds\n"
            if 'label' in labeled_df.columns:
                summary_log += "Overall Behavioral Label Distribution:\n"
                for k, v in label_counts.set_index('label')['count'].to_dict().items():
                    summary_log += f"    {k}: {v} instances\n"
            progress_queue.put(("log", summary_log))

            report_path = os.path.join(reports_dir, 'report.html')
            write_html_report(report_path, stats_all, stats_c, labeled_df, processed_df, summary_img_path, viz_dir, config)
            monitor.report_path = report_path
            progress_queue.put(("done", None))
        except Exception as e:
            progress_queue.put(("error", f"An error occurred: {e}"))
        finally:
            # Restore stdout
            pass
    # --- Run Pipeline ---
    pipeline_thread = threading.Thread(target=pipeline_logic, args=(progress_queue, cancel_event))
    monitor.pipeline_thread = pipeline_thread  # Pass the thread to the monitor for access in _on_closing
    pipeline_thread.start()
    
    monitor.start_timer()
    progress_root.after(100, process_queue)
    progress_root.mainloop()
    
    pipeline_thread.join()
    pipeline_thread.join()

    # Restore stdout
    sys.stdout = original_stdout

if __name__ == "__main__":
    main()

print("--- Script execution finished ---")
