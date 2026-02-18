"""
Simple GUI for Eye-Tracking Deep Learning Models
================================================
User-friendly interface to run MLP, CNN, CNN1D, LSTM, BiLSTM, Hybrid, and Transformer models.
"""

import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import webbrowser
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config_loader import load_config

# Lazy import for heavy libraries (loaded on first run)
_pipeline_module = None
_config = None

def _check_dll_access():
    """
    Pre-check: Test if critical DLL files can be loaded.
    If not, restart with admin privileges automatically.
    
    CURRENTLY DISABLED: Direct testing works but this pre-check fails.
    This suggests environment/timing issue. Let GUI load naturally.
    """
    # Skip pre-check - let GUI handle DLL issues during normal operation
    return True

def _get_config():
    """Load configuration if not already loaded."""
    global _config
    if _config is None:
        _config = load_config()
    return _config

def _restart_as_admin():
    """Restart the GUI with administrator privileges."""
    import sys
    import os
    import subprocess
    import ctypes
    
    # Check if already running as admin
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        is_admin = False
    
    if is_admin:
        # Already admin but still got DLL error - show error
        print("\n[ERROR] Already running as Administrator but DLL load still fails!")
        print("This indicates a system-level restriction that cannot be bypassed.\n")
        return
    
    # Get the Python executable and script path
    python_exe = sys.executable
    script_path = os.path.abspath(__file__)
    
    # Use PowerShell to restart with elevation
    ps_command = (
        f'Start-Process -FilePath "{python_exe}" '
        f'-ArgumentList "{script_path}" '
        f'-Verb RunAs -WorkingDirectory "{os.path.dirname(script_path)}"'
    )
    
    try:
        print("\n[INFO] Requesting Windows UAC elevation...")
        print("[INFO] A UAC prompt will appear - Click 'Yes' to continue.\n")
        
        # Try to show GUI message if Tk is available
        try:
            import tkinter as tk
            import tkinter.messagebox as mb
            
            # Check if root exists
            root = tk._default_root
            if root:
                mb.showinfo(
                    "Requesting Administrator Access",
                    "Windows will ask for Administrator permission.\n\n"
                    "Click 'Yes' to continue."
                )
                root.destroy()
            else:
                # No root yet - create temporary one
                temp_root = tk.Tk()
                temp_root.withdraw()
                mb.showinfo(
                    "Requesting Administrator Access",
                    "Windows will ask for Administrator permission.\n\n"
                    "Click 'Yes' to continue."
                )
                temp_root.destroy()
        except:
            # GUI not available or failed - continue without message
            pass
        
        # Launch with admin
        subprocess.Popen(['powershell', '-Command', ps_command], 
                        creationflags=subprocess.CREATE_NO_WINDOW)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Failed to restart with admin: {e}")
        sys.exit(1)

def _get_pipeline_module():
    """Dynamic import of pipeline module to ensure latest code is used."""
    import sys
    try:
        # Force remove ALL relevant modules from sys.modules to guarantee fresh load
        modules_to_remove = [
            k for k in sys.modules 
            if any(x in k for x in [
                'run_pipeline_simple', 
                'config_loader', 
                'simple_trainer',
                'reporting', 
                'src.train',
                'src.reporting',
                'src.utils'
            ])
        ]
        for k in modules_to_remove:
            del sys.modules[k]
        
        print("[DEBUG] Cleared modules:", len(modules_to_remove), "modules removed from cache")
            
        import run_pipeline_simple
        from run_pipeline_simple import run_simple_pipeline, run_all_models, run_pipeline_with_lopo_cv
        return {
            'run_simple_pipeline': run_simple_pipeline, 
            'run_all_models': run_all_models,
            'run_lopo_cv': run_pipeline_with_lopo_cv  # Alias for consistency
        }
    except Exception as e:
        # If we reach here after pre-check passed, something else is wrong
        print(f"\n[ERROR] Failed to import pipeline module: {e}")
        import traceback
        traceback.print_exc()
        raise


class SimpleGUI:
    """Simple GUI for model training."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Eye-Tracking Deep Learning - Simple Models")
        
        # Check if running with admin privileges (Windows)
        self.is_admin = self._check_admin_windows()
        
        # Load config
        self.config = _get_config()
        
        # Window setup
        w, h = 920, 820
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.root.minsize(920, 820)  # Prevent resizing below this size
        self.root.configure(bg='#f5f5f5')
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self.is_running = False
        self.last_result = None
        
        self._build_ui()
        
        # Initialize UI state based on evaluation method
        self._on_eval_method_change()
    
    def _check_admin_windows(self):
        """Check if running with administrator privileges on Windows."""
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False
    
    def _on_closing(self):
        """Handle window close event."""
        if self.is_running:
            if messagebox.askokcancel("Quit", "Training is in progress. Do you want to quit?"):
                self.root.destroy()
        else:
            self.root.destroy()
    
    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg='white', height=50)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text="Eye-Tracking Deep Learning", 
                font=('Segoe UI', 14, 'bold'),
                bg='white', fg='#212121').pack(side='left', padx=20, pady=10)
        
        # Status indicator for admin privileges
        if self.is_admin:
            status_frame = tk.Frame(self.root, bg='#d4edda', height=35)
            status_frame.pack(fill='x')
            status_frame.pack_propagate(False)
            
            tk.Label(status_frame, text="[OK] Running with Administrator privileges", 
                    bg='#d4edda', fg='#155724', 
                    font=('Segoe UI', 9)).pack(pady=8)
        
        # Main container
        main = tk.Frame(self.root, bg='#f5f5f5')
        main.pack(fill='both', expand=True, padx=12, pady=12)
        
        # Left panel: Configuration
        left = tk.Frame(main, bg='white', relief='solid', bd=1)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.configure(width=350)
        
        tk.Label(left, text="Configuration", font=('Segoe UI', 11, 'bold'),
                bg='white', fg='#212121').pack(pady=8, padx=15, anchor='w')

        # Buttons frame (Packed first to ensure it stays at bottom)
        btn_frame = tk.Frame(left, bg='white')
        btn_frame.pack(side='bottom', fill='x', padx=15, pady=10)
        
        config = tk.Frame(left, bg='white')
        config.pack(fill='both', expand=True, padx=15, pady=3)
        
        # Model Type
        row = 0
        tk.Label(config, text="Model Type", font=('Segoe UI', 9, 'bold'),
                bg='white', fg='#424242').grid(row=row, column=0, sticky='w', pady=(5, 3))
        row += 1
        
        self.model_var = tk.StringVar(value='lstm')  # Default to LSTM (best model)
        model_combo = ttk.Combobox(config, textvariable=self.model_var,
                                   values=['mlp', 'cnn', 'cnn1d', 'lstm', 'bilstm', 'hybrid', 'transformer'],
                                   state='readonly', font=('Segoe UI', 9), width=30)
        model_combo.grid(row=row, column=0, sticky='ew', pady=(0, 10))
        row += 1
        
        # Parts Filter
        tk.Label(config, text="Trial Parts", font=('Segoe UI', 9, 'bold'),
                bg='white', fg='#424242').grid(row=row, column=0, sticky='w', pady=(5, 3))
        row += 1
        
        self.parts_var = tk.StringVar(value='timer_only')
        parts_frame = tk.Frame(config, bg='white')
        parts_frame.grid(row=row, column=0, sticky='w', pady=(0, 10))
        row += 1
        
        tk.Radiobutton(parts_frame, text="Timer Trials", 
                      variable=self.parts_var, value='timer_only',
                      font=('Segoe UI', 8), bg='white').pack(anchor='w', pady=1)
        tk.Radiobutton(parts_frame, text="No-Timer + Timer-No-Correct", 
                      variable=self.parts_var, value='incorrect',
                      font=('Segoe UI', 8), bg='white').pack(anchor='w', pady=1)
        tk.Radiobutton(parts_frame, text="All Parts", 
                      variable=self.parts_var, value='all',
                      font=('Segoe UI', 8), bg='white').pack(anchor='w', pady=1)
        
        # AOI Filter
        tk.Label(config, text="Area of Interest (AOI)", font=('Segoe UI', 9, 'bold'),
                bg='white', fg='#424242').grid(row=row, column=0, sticky='w', pady=(5, 3))
        row += 1
        
        self.aoi_var = tk.StringVar(value='answer')
        aoi_frame = tk.Frame(config, bg='white')
        aoi_frame.grid(row=row, column=0, sticky='w', pady=(0, 10))
        row += 1
        
        tk.Radiobutton(aoi_frame, text="Answer Area", 
                      variable=self.aoi_var, value='answer',
                      font=('Segoe UI', 8), bg='white').pack(anchor='w', pady=1)
        tk.Radiobutton(aoi_frame, text="All AOIs", 
                      variable=self.aoi_var, value='all_aois',
                      font=('Segoe UI', 8), bg='white').pack(anchor='w', pady=1)
        
        # Time Window
        tk.Label(config, text="Time Window", font=('Segoe UI', 9, 'bold'),
                bg='white', fg='#424242').grid(row=row, column=0, sticky='w', pady=(5, 3))
        row += 1
        
        # Load default time window from config
        default_time = self.config.data['time_window_s'] or 8
        self.time_var = tk.IntVar(value=default_time)
        time_frame = tk.Frame(config, bg='white')
        time_frame.grid(row=row, column=0, sticky='w', pady=(0, 3))
        row += 1
        
        self.time_label_var = tk.StringVar(value=f"Last {default_time} seconds")
        tk.Label(time_frame, textvariable=self.time_label_var, 
                font=('Segoe UI', 10, 'bold'), bg='white', fg='#2196F3').pack(side='left')
        row_time = row
        
        time_slider = tk.Scale(config, variable=self.time_var, from_=4, to=12, orient='horizontal',
                              length=250, font=('Segoe UI', 8), bg='white',
                              troughcolor='#e0e0e0', highlightthickness=0, showvalue=0)
        time_slider.grid(row=row, column=0, pady=(0, 3))
        row += 1
        
        self.full_time_var = tk.BooleanVar(value=False)
        tk.Checkbutton(config, text="Use full trial duration", 
                      variable=self.full_time_var,
                      font=('Segoe UI', 8), bg='white', fg='#424242',
                      command=self._toggle_time_slider).grid(row=row, column=0, sticky='w', pady=(0, 10))
        row += 1
        
        def update_time_label(*args):
            if self.full_time_var.get():
                self.time_label_var.set("Full trial")
            else:
                self.time_label_var.set(f"Last {self.time_var.get()} seconds")
        self.time_var.trace('w', update_time_label)
        self.full_time_var.trace('w', update_time_label)
        
        # Feature Selection
        tk.Label(config, text="Feature Selection", font=('Segoe UI', 9, 'bold'),
                bg='white', fg='#424242').grid(row=row, column=0, sticky='w', pady=(5, 3))
        row += 1
        
        self.features_var = tk.StringVar(value='selected')
        features_frame = tk.Frame(config, bg='white')
        features_frame.grid(row=row, column=0, sticky='w', pady=(0, 10))
        row += 1
        
        tk.Radiobutton(features_frame, text="Selected 13 Features", 
                      variable=self.features_var, value='selected',
                      font=('Segoe UI', 8), bg='white').pack(anchor='w', pady=1)
        tk.Radiobutton(features_frame, text="All Eye-Tracking Features", 
                      variable=self.features_var, value='all',
                      font=('Segoe UI', 8), bg='white').pack(anchor='w', pady=1)
        
        # Evaluation Method
        tk.Label(config, text="Evaluation Method", font=('Segoe UI', 9, 'bold'),
                bg='white', fg='#424242').grid(row=row, column=0, sticky='w', pady=(5, 3))
        row += 1
        
        self.eval_method_var = tk.StringVar(value='lopo')
        eval_frame = tk.Frame(config, bg='white')
        eval_frame.grid(row=row, column=0, sticky='w', pady=(0, 10))
        row += 1
        
        tk.Radiobutton(eval_frame, text="LOPO", 
                      variable=self.eval_method_var, value='lopo',
                      font=('Segoe UI', 8), bg='white',
                      command=self._on_eval_method_change).pack(anchor='w', pady=1)
        tk.Radiobutton(eval_frame, text="Simple", 
                      variable=self.eval_method_var, value='simple',
                      font=('Segoe UI', 8), bg='white',
                      command=self._on_eval_method_change).pack(anchor='w', pady=1)
        
        # Buttons - Fixed at bottom of left panel
        
        self.run_btn = tk.Button(btn_frame, text="> Run Model", 
                                command=self._run_pipeline,
                                font=('Segoe UI', 10, 'bold'), bg='#4CAF50', fg='white',
                                relief='flat', bd=0, cursor='hand2', height=2)
        self.run_btn.pack(fill='x', pady=(0, 6))
        
        self.run_all_btn = tk.Button(btn_frame, text=">> Run All Models", 
                                     command=self._run_all_models,
                                     font=('Segoe UI', 9, 'bold'), bg='#2196F3', fg='white',
                                     relief='flat', bd=0, cursor='hand2', height=2)
        self.run_all_btn.pack(fill='x', pady=(0, 6))
        
        self.report_btn = tk.Button(btn_frame, text="Open Dashboard", command=self._open_report,
                 font=('Segoe UI', 9), bg='#546E7A', fg='white',
                 relief='flat', bd=0, cursor='hand2', height=1)
        self.report_btn.pack(fill='x')
        
        # Right panel: Log
        right = tk.Frame(main, bg='white', relief='solid', bd=1)
        right.pack(side='right', fill='both', expand=True)
        
        tk.Label(right, text="Training Log", font=('Segoe UI', 11, 'bold'),
                bg='white', fg='#212121').pack(pady=8, padx=15, anchor='w')
        
        self.log_text = scrolledtext.ScrolledText(right, font=('Consolas', 9),
                                                  bg='#263238', fg='#00FF00',
                                                  relief='flat', bd=0, wrap='word')
        self.log_text.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Add right-click context menu for copy
        self.context_menu = tk.Menu(self.log_text, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self._copy_selection)
        self.context_menu.add_command(label="Copy All", command=self._copy_all)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Clear", command=self._clear_log)
        
        def show_context_menu(event):
            self.context_menu.post(event.x_root, event.y_root)
        
        self.log_text.bind("<Button-3>", show_context_menu)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status = tk.Label(self.root, textvariable=self.status_var, 
                         font=('Segoe UI', 8), bg='#eeeeee', fg='#424242',
                         anchor='w', relief='flat', bd=1)
        status.pack(side='bottom', fill='x', padx=0, pady=0, ipady=3)
    
    def _toggle_time_slider(self):
        """Toggle time slider based on checkbox."""
        pass  # Handled by trace
    
    def _on_eval_method_change(self):
        """Handle evaluation method change - disable Run All Models for LOPO."""
        if self.eval_method_var.get() == 'lopo':
            # Disable "Run All Models" button (LOPO only supports single model)
            self.run_all_btn.config(state='disabled', bg='#cccccc')
        else:
            # Enable "Run All Models" button for simple split
            if not self.is_running:
                self.run_all_btn.config(state='normal', bg='#2196F3')
    
    def _get_parts_filter(self):
        """Get parts filter based on selection."""
        parts_map = {
            'timer_only': ['Timer-Correct', 'Timer-No-Correct'],
            'incorrect': ['No-Timer', 'Timer-No-Correct'],
            'all': []  # Use empty list to signal "All Parts" (avoiding None which triggers defaults)
        }
        return parts_map.get(self.parts_var.get(), [])
    
    def _get_aoi_filter(self):
        """Get AOI filter based on selection."""
        aoi_map = {
            'answer': ['Answer_Area'],
            'all_aois': [],  # Use empty list to signal "All AOIs"
            'all': []        # Backward compatibility
        }
        val = self.aoi_var.get()
        print(f"DEBUG: GUI AOI Var: {val}") # Console print
        return aoi_map.get(val, [])
    
    def _get_feature_columns(self):
        """Get feature columns based on selection."""
        if self.features_var.get() == 'selected':
            # 13 selected high-quality features (based on actual column names in trial CSVs)
            return ['BPOGX', 'BPOGY', 'FPOGD', 'FPOGX', 'FPOGY', 
                    'LPCX', 'LPCY', 'LPD', 'LPUPILD', 'RPCX', 'RPCY', 'RPD', 'RPUPILD']
        else:
            # Use all available features
            return None
    
    def _get_trials_dir(self):
        """Get trials directory."""
        return r"D:\MasterThesis\MasterThesis\DataMining\results\reports\final\trials"
    
    def _log(self, message):
        """Add message to log and console."""
        print(message, flush=True)  # Print to console
        self.log_text.insert('end', message + '\n')
        self.log_text.see('end')
        self.root.update()
    
    def _copy_selection(self):
        """Copy selected text to clipboard."""
        try:
            selected = self.log_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected)
            self.status_var.set("[OK] Selection copied!")
        except tk.TclError:
            messagebox.showinfo("No Selection", "No text selected.")
    
    def _copy_all(self):
        """Copy all log text to clipboard."""
        log_content = self.log_text.get('1.0', 'end-1c')
        if log_content.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(log_content)
            self.status_var.set("[OK] All text copied!")
        else:
            messagebox.showinfo("Empty", "No text to copy.")
    
    def _clear_log(self):
        """Clear log text."""
        if messagebox.askyesno("Clear Log", "Clear all log text?"):
            self.log_text.delete('1.0', 'end')
            self.status_var.set("Log cleared")
    
    def _run_pipeline(self):
        """Run single model pipeline."""
        if self.is_running:
            messagebox.showwarning("Running", "Pipeline is already running!")
            return
        
        # Get configuration
        model_type = self.model_var.get()
        parts_filter = self._get_parts_filter()
        aoi_filter = self._get_aoi_filter()
        # Use -1 for full trial (not None, to avoid config override)
        time_window = -1 if self.full_time_var.get() else self.time_var.get()
        feature_cols = self._get_feature_columns()
        trials_dir = self._get_trials_dir()
        
        # Run in thread
        def run():
            self.is_running = True
            self.status_var.set(f"Loading libraries and running {model_type.upper()}...")
            self.run_btn.config(state='disabled', bg='#999999')
            self.run_all_btn.config(state='disabled', bg='#999999')
            self.log_text.delete('1.0', 'end')
            
            # Log loading message
            self._log("Loading ML libraries (this may take a few seconds on first run)...")
            self._log(f"DEBUG: Selected Parts: {parts_filter}")
            self._log(f"DEBUG: Selected AOI: {aoi_filter}")
            self.root.update()
            
            try:
                # TeeOutput to capture print statements and show in GUI
                class TeeOutput:
                    def __init__(self, gui_log_func, original_stdout):
                        self.gui_log = gui_log_func
                        self.stdout = original_stdout
                        self.buffer = []
                    
                    def write(self, text):
                        self.stdout.write(text)  # Write to console
                        self.stdout.flush()
                        self.buffer.append(text)
                        if '\n' in text:
                            line = ''.join(self.buffer).rstrip('\n')
                            if line:
                                self.root.after(0, lambda l=line: self.gui_log(l))
                            self.buffer = []
                    
                    def flush(self):
                        self.stdout.flush()
                
                import sys
                old_stdout = sys.stdout
                tee = TeeOutput(lambda txt: self.log_text.insert('end', txt + '\n') or self.log_text.see('end') or self.root.update(), old_stdout)
                tee.root = self.root
                sys.stdout = tee
                
                # Lazy import pipeline functions
                pipeline = _get_pipeline_module()
                
                # Log completion
                self.log_text.insert('end', "[OK] Libraries loaded successfully!\n\n")
                self.log_text.see('end')
                self.root.update()
                
                # Check evaluation method
                eval_method = self.eval_method_var.get()
                
                if eval_method == 'lopo':
                    self.log_text.insert('end', "Run mode: LOPO Cross-Validation (Participant-Independent)...\n")
                    run_lopo_cv = pipeline['run_lopo_cv']
                    result = run_lopo_cv(
                        model_type=model_type,
                        parts_filter=parts_filter,
                        aoi_filter=aoi_filter,
                        time_window_s=time_window,
                        feature_columns=feature_cols,
                        n_seeds=1  # Default 1 seed for GUI (faster)
                    )
                    
                    # LOPO returns metrics directly
                    accuracy = result['metrics']['accuracy']
                    f1 = result['metrics']['f1']
                    auc = result['metrics']['roc_auc']
                    
                else:  # simple split
                    self.log_text.insert('end', "[WARN] Run mode: Simple Train/Test Split (MAY have participant leakage)...\n")
                    run_simple_pipeline = pipeline['run_simple_pipeline']
                    result = run_simple_pipeline(
                        model_type=model_type,
                        parts_filter=parts_filter,
                        aoi_filter=aoi_filter,
                        time_window_s=time_window,
                        feature_columns=feature_cols,
                        trials_dir=trials_dir
                    )
                    
                    accuracy = result['results']['test']['accuracy']
                    f1 = result['results']['test']['f1']
                    auc = result['results']['test']['roc_auc']
                
                self.status_var.set(f"[OK] Complete! Acc={accuracy*100:.1f}%, AUC={auc:.3f}")
                success_msg = f"Training complete!\n\n"
                success_msg += f"Test Accuracy: {accuracy*100:.2f}%\n"
                success_msg += f"Test AUC: {auc:.4f}\n"
                success_msg += f"Test F1: {f1:.4f}\n\n"
                
                if result.get('html_path'):
                    success_msg += f"HTML Report: {Path(result['html_path']).name}\n"
                
                sys.stdout = old_stdout
                self.last_result = result
                messagebox.showinfo("Success", success_msg)
                
            except Exception as e:
                sys.stdout = old_stdout
                import traceback
                error_details = traceback.format_exc()
                self.log_text.insert('end', f"\n{'='*70}\n")
                self.log_text.insert('end', "ERROR DETAILS:\n")
                self.log_text.insert('end', error_details)
                self.log_text.insert('end', f"{'='*70}\n")
                self.log_text.see('end')
                self.status_var.set(f"[FAIL] Error: {str(e)[:50]}")
                messagebox.showerror("Error", f"Pipeline failed:\n{str(e)}")
            
            finally:
                self.is_running = False
                self.run_btn.config(state='normal', bg='#4CAF50')
                self.run_all_btn.config(state='normal', bg='#2196F3')
        
        threading.Thread(target=run, daemon=True).start()
    
    def _run_all_models(self):
        """Run all models."""
        if self.is_running:
            messagebox.showwarning("Running", "Pipeline is already running!")
            return
        
        if not messagebox.askyesno("Confirm", 
            "Run all 7 models (MLP, CNN, CNN1D, LSTM, BiLSTM, Hybrid, Transformer)?\nThis may take several minutes."):
            return
        
        # Get configuration
        parts_filter = self._get_parts_filter()
        aoi_filter = self._get_aoi_filter()
        time_window = None if self.full_time_var.get() else self.time_var.get()
        feature_cols = self._get_feature_columns()
        trials_dir = self._get_trials_dir()
        
        # Run in thread
        def run():
            self.is_running = True
            self.status_var.set("Loading libraries and running all models...")
            self.run_btn.config(state='disabled', bg='#999999')
            self.run_all_btn.config(state='disabled', bg='#999999')
            self.log_text.delete('1.0', 'end')
            
            # Log loading message
            self._log("Loading ML libraries (this may take a few seconds on first run)...")
            self.root.update()
            
            try:
                # Capture output to both GUI and console
                class TeeOutput:
                    def __init__(self, gui_log_func, original_stdout):
                        self.gui_log = gui_log_func
                        self.stdout = original_stdout
                        self.buffer = []
                    
                    def write(self, text):
                        self.stdout.write(text)  # Write to console
                        self.stdout.flush()
                        self.buffer.append(text)
                        if '\n' in text:
                            line = ''.join(self.buffer).rstrip('\n')
                            if line:
                                self.root.after(0, lambda l=line: self.gui_log(l))
                            self.buffer = []
                    
                    def flush(self):
                        self.stdout.flush()
                
                import sys
                old_stdout = sys.stdout
                tee = TeeOutput(lambda txt: self.log_text.insert('end', txt + '\n') or self.log_text.see('end') or self.root.update(), old_stdout)
                tee.root = self.root
                sys.stdout = tee
                
                # Lazy import pipeline functions
                pipeline = _get_pipeline_module()
                run_all_models = pipeline['run_all_models']
                
                # Log completion
                self.log_text.insert('end', "[OK] Libraries loaded successfully!\n\n")
                self.log_text.see('end')
                self.root.update()
                
                results = run_all_models(
                    parts_filter=parts_filter,
                    aoi_filter=aoi_filter,
                    time_window_s=time_window,
                    feature_columns=feature_cols,
                    trials_dir=trials_dir
                )
                
                sys.stdout = old_stdout
                self.last_result = results
                
                self.status_var.set("[OK] All models complete!")
                messagebox.showinfo("Success", "All models trained successfully!")
                
            except Exception as e:
                # IMPORTANT: If CUDA is broken, we MUST restart
                is_cuda_error = "CUDA" in str(e) or "illegal instruction" in str(e) or "AcceleratorError" in str(e)
                
                sys.stdout = old_stdout
                import traceback
                error_details = traceback.format_exc()
                self.log_text.insert('end', f"\n{'='*70}\n")
                self.log_text.insert('end', "ERROR DETAILS:\n")
                self.log_text.insert('end', error_details)
                self.log_text.insert('end', f"{'='*70}\n")
                
                if is_cuda_error:
                    self.log_text.insert('end', "\n!!! CRITICAL CUDA ERROR !!!\n")
                    self.log_text.insert('end', "Your GPU driver state is corrupted (likely from a previous crash).\n")
                    self.log_text.insert('end', "You MUST close and restart this application completely to fix it.\n")
                    self.log_text.insert('end', "Clicking Run again will NOT work.\n")
                    self.log_text.see('end')
                    messagebox.showerror("Critical CUDA Error", 
                        "GPU Critical Error Detected!\n\n"
                        "The graphics driver state is corrupted.\n"
                        "You MUST CLOSE and RESTART this application completely.\n\n"
                        "Running again without restart will fail.")
                else:
                    self.log_text.see('end')
                    self.status_var.set(f"[FAIL] Error")
                    messagebox.showerror("Error", f"Pipeline failed:\n{str(e)}")
            
            finally:
                self.is_running = False
                self.run_btn.config(state='normal', bg='#4CAF50')
                self.run_all_btn.config(state='normal', bg='#2196F3')
        
        threading.Thread(target=run, daemon=True).start()
    
    def _open_report(self):
        """Open main dashboard (index.html)."""
        # Always open the main dashboard
        # This path must be absolute to avoid issues with CWD
        project_root = Path(r"D:\MasterThesis\MasterThesis\DeepLearning")
        reports_dir = project_root / 'outputs' / 'reports'
        dashboard_path = reports_dir / 'index.html'

        if dashboard_path.exists():
            webbrowser.open(f'file:///{str(dashboard_path)}')
            self.status_var.set(f"[OK] Opened Dashboard")
        else:
            # If dashboard doesn't exist, try to generate it
            try:
                # Ensure src is in path to import AdvancedReportGenerator
                sys.path.insert(0, str(project_root / 'src'))
                
                # Force remove reporting modules to ensure freshness
                modules_to_remove = [k for k in sys.modules if k.startswith('reporting')]
                for k in modules_to_remove:
                    del sys.modules[k]
                
                from reporting.html_report import AdvancedReportGenerator
                
                reporter = AdvancedReportGenerator(reports_dir)
                reporter.generate_dashboard()
                
                webbrowser.open(f'file:///{str(dashboard_path)}')
                self.status_var.set(f"[OK] Generated and opened Dashboard")
            except Exception as e:
                messagebox.showinfo("No Reports", 
                    f"Dashboard not available yet. Run a model first!\n\nPath checked: {dashboard_path}\nError: {str(e)}")

def main():
    # Pre-check: Ensure DLL files can be loaded (auto-restart with admin if needed)
    print("Checking DLL access...")
    if not _check_dll_access():
        # Restarting with admin - exit this instance
        return
    
    print("[OK] DLL access verified\n")
    
    root = tk.Tk()
    app = SimpleGUI(root)
    
    # Force window to update and stabilize
    root.update_idletasks()
    root.update()
    
    # Start mainloop
    root.mainloop()

if __name__ == '__main__':
    main()

