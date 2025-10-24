# %% [Section 1]: Imports & Configuration
import os
import json
import time
import socket
import re
import csv
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import configparser
import shutil

# Load configuration from config.ini using an absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.ini')
config = configparser.ConfigParser()
config.read(config_path)

# Centralized configuration dictionary
CONFIG = {
    'SCREEN': {k: config.getboolean('SCREEN', k) if k in ('fullscreen', 'topmost') else config.getint('SCREEN', k) 
               for k in config['SCREEN']},
    'PATHS': dict(config['PATHS']),
    'EYE_TRACKER': {
        'host': config.get('EYE_TRACKER', 'host'),
        'port': config.getint('EYE_TRACKER', 'port'),
        'commands': [cmd.strip() for cmd in config.get('EYE_TRACKER', 'commands').split(',')],
    },
    'DIMENSIONS': {k: config.getfloat('DIMENSIONS', k) if 'ratio' in k else config.getint('DIMENSIONS', k) 
                   for k in config['DIMENSIONS']},
    'COLORS': dict(config['COLORS']),
    'FONTS': {k: config.getint('FONTS', k) if 'size' in k and 'ratio' not in k else 
              config.getfloat('FONTS', k) if 'ratio' in k else config.get('FONTS', k) 
              for k in config['FONTS']},
    'TIMING': {k: config.getint('TIMING', k) for k in config['TIMING']},
}

# %% [Section 2]: Eye Tracker Client
class EyeTrackerClient:
    """Handles communication with the eye-tracking device."""
    
    def __init__(self):
        """Initialize the eye tracker client with default attributes."""
        self.socket = None
        self.receive_thread = None
        self.should_stop = False
        self.data_records = []
        self.csv_filepath = None

    def connect(self):
        """Establish a connection to the eye tracker using socket."""
        # --- Commented out eye tracker connection code ---
        # try:
        #     self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #     self.socket.connect((CONFIG['EYE_TRACKER']['host'], CONFIG['EYE_TRACKER']['port']))
        #     for command in CONFIG['EYE_TRACKER']['commands']:
        #         self.socket.send(command.encode('utf-8'))
        #         self.socket.recv(1024)
        # except Exception as e:
        #     raise ConnectionError(f"Failed to connect to eye tracker: {e}")
        pass

    def start_recording(self, csv_filepath):
        """Start recording eye-tracking data to the specified CSV file."""
        self.data_records.clear()
        self.csv_filepath = csv_filepath
        # --- Commented out eye tracker start recording code ---
        # self.socket.send(b'<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n')
        # self.should_stop = False
        # self.receive_thread = threading.Thread(target=self._receive_data, daemon=True)
        # self.receive_thread.start()
        pass

    def stop_recording(self):
        """Stop recording eye-tracking data and save it to CSV."""
        # --- Commented out eye tracker stop recording code ---
        # self.socket.send(b'<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n')
        # self.should_stop = True
        # if self.receive_thread:
        #     self.receive_thread.join(timeout=2.0)
        # self._save_to_csv()
        pass

    def _receive_data(self):
        """Background thread to continuously receive data from the eye tracker."""
        # --- Commented out eye tracker data reception code ---
        # buffer = ""
        # while not self.should_stop:
        #     try:
        #         chunk = self.socket.recv(2048).decode('utf-8', errors='ignore')
        #         buffer += chunk
        #         lines = buffer.split('\r\n')
        #         buffer = lines[-1]
        #         for line in lines[:-1]:
        #             if line.startswith("<REC"):
        #                 self.data_records.append(line.strip())
        #     except Exception as e:
        #         if not self.should_stop:
        #             messagebox.showerror("Data Error", f"Error receiving eye-tracking data: {e}")
        #         break
        pass

    def _save_to_csv(self):
        """Parse recorded data and save it to a CSV file."""
        if not self.data_records:
            return
        attr_pattern = re.compile(r'(\w+)="([^"]*)"')
        keys = set()
        parsed_data = []
        for record in self.data_records:
            attributes = dict(attr_pattern.findall(record))
            keys.update(attributes.keys())
            parsed_data.append(attributes)
        with open(self.csv_filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=sorted(keys))
            writer.writeheader()
            writer.writerows(parsed_data)

    def close(self):
        """Close the eye tracker socket connection."""
        if self.socket:
            self.socket.close()

# %% [Section 3]: Main Application
class QuestionDisplayApp:
    """Core application class managing the responsive UI and question flow."""
    
    def __init__(self, root, participant_id):
        """Initialize the application with root window and participant details."""
        self.root = root
        self.participant_id = participant_id
        self.eye_tracker = EyeTrackerClient()
        self.question_list = []
        self.question_bank = {}
        self.current_index = 0
        self.question_number = 1
        self.part1_question_count = 0
        self.answers = []
        self.timer_id = None
        self.blink_id = None
        self.auto_timer = None
        self.selected_option_id = None
        
        # Define participant-specific directory under Output
        self.participant_dir = os.path.join(CONFIG['PATHS']['output_dir'], f"participant_{self.participant_id}")
        
        # Check and create directory structure
        if not self._setup_participant_directory():
            self.root.destroy()
            return

        # Get initial screen dimensions
        self.screen_width = CONFIG['SCREEN']['width'] or root.winfo_screenwidth()
        self.screen_height = CONFIG['SCREEN']['height'] or root.winfo_screenheight()

        # Load layout ratios from config
        self.left_panel_ratio = CONFIG['DIMENSIONS']['left_panel_ratio']
        self.right_panel_ratio = CONFIG['DIMENSIONS']['right_panel_ratio']
        self.question_section_ratio = CONFIG['DIMENSIONS']['question_section_ratio']
        self.options_section_ratio = CONFIG['DIMENSIONS']['options_section_ratio']

        # Initialize UI and data
        self._setup_window()
        # Load questions; if loading fails, stop initialization to avoid operating on a destroyed root
        if not self._load_questions():
            return
        self._build_ui()
        self._connect_eye_tracker()
        self.display_next()

    def _setup_participant_directory(self):
        """Setup participant directory and handle existing folder case."""
        output_dir = CONFIG['PATHS']['output_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if os.path.exists(self.participant_dir):
            response = messagebox.askyesno(
                "Directory Exists",
                f"Participant directory '{self.participant_dir}' already exists.\n"
                "Would you like to overwrite existing data?"
            )
            if not response:
                messagebox.showinfo("Program Terminated", "Program will exit without overwriting existing data.")
                return False
            shutil.rmtree(self.participant_dir)
        
        os.makedirs(self.participant_dir, exist_ok=True)
        return True

    def _setup_window(self):
        """Configure the main window with fixed dimensions and key bindings."""
        self.root.title("Exam Interface")
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")
        self.root.configure(bg=CONFIG['COLORS']['background'])
        self.root.attributes('-fullscreen', CONFIG['SCREEN']['fullscreen'])
        self.root.attributes('-topmost', CONFIG['SCREEN']['topmost'])
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=int(self.left_panel_ratio * 10))
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=int(self.right_panel_ratio * 10))
        
        self.root.bind('<Escape>', lambda e: self._exit_application())
        self.root.bind('<Configure>', self._on_resize)

    def _build_ui(self):
        """Build the UI components with responsive sizes."""
        self._update_dimensions()
        self.question_font_size = max(int(((self.left_width + self.screen_height) / 2) * CONFIG['FONTS']['question_size_ratio']), 8)
        self._create_panels()
        self._configure_sizes()
    def _update_dimensions(self):
        """Update panel dimensions based on current window size and config ratios."""
        self.screen_width = self.root.winfo_width()
        self.screen_height = self.root.winfo_height()
        self.left_width = int(self.screen_width * self.left_panel_ratio)
        self.right_width = int(self.screen_width * self.right_panel_ratio)
        self.question_height = int(self.screen_height * self.question_section_ratio)
        self.options_height = int(self.screen_height * self.options_section_ratio)

    def _create_panels(self):
        """Create UI panels with fixed sizes and configurable padding."""
        # Left Panel
        if hasattr(self, 'left_panel'):
            self.left_panel.destroy()
        self.left_panel = tk.Frame(self.root, bg=CONFIG['COLORS']['background'], 
                                   width=self.left_width, height=self.screen_height)
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        self.left_panel.grid_propagate(False)
        self.left_panel.grid_rowconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=1)
        self.left_panel.grid_rowconfigure(2, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)

        # Question Frame
        if hasattr(self, 'question_frame'):
            self.question_frame.destroy()
        self.question_frame = tk.Frame(self.left_panel, bg=CONFIG['COLORS']['question_bg'], 
                                       height=self.question_height)
        self.question_frame.grid(row=0, column=0, sticky="nsew")
        self.question_frame.grid_propagate(False)
        self.question_inner = tk.Frame(self.question_frame, bg=CONFIG['COLORS']['question_bg'])
        self.question_inner.grid(row=0, column=0, sticky="nsew",
                                 padx=int(self.left_width * CONFIG['DIMENSIONS']['question_inner_padx_ratio']),
                                 pady=int(self.question_height * CONFIG['DIMENSIONS']['question_inner_pady_ratio']))
        self.question_text = tk.Message(self.question_inner,
                                width=int(self.left_width * 0.9),
                                bg=CONFIG['COLORS']['question_bg'],
                                font=(CONFIG['FONTS']['family'], self.question_font_size),
                                justify="left")
        self.question_text.grid(row=0, column=0, sticky="nsew")
        self.question_frame.grid_rowconfigure(0, weight=1)
        self.question_frame.grid_columnconfigure(0, weight=1)

        # Left Gap Frame
        if hasattr(self, 'left_gap_frame'):
            self.left_gap_frame.destroy()
        self.left_gap_frame = tk.Frame(self.left_panel, bg=CONFIG['COLORS']['background'], 
                                      height=self.screen_height * CONFIG['DIMENSIONS']['left_frame_gap_height_ratio'])
        self.left_gap_frame.grid(row=1, column=0, sticky="nsew",
                                padx=int(self.left_width * CONFIG['DIMENSIONS']['options_frame_padx_ratio']),
                                pady=int(self.options_height * CONFIG['DIMENSIONS']['options_frame_pady_ratio']))
        self.left_gap_frame.grid_propagate(False)

        # Options Frame
        if hasattr(self, 'options_frame'):
            self.options_frame.destroy()
        self.options_frame = tk.Frame(self.left_panel, bg=CONFIG['COLORS']['background'], 
                                      height=self.options_height)
        self.options_frame.grid(row=2, column=0, sticky="nsew",
                                padx=int(self.left_width * CONFIG['DIMENSIONS']['options_frame_padx_ratio']),
                                pady=int(self.options_height * CONFIG['DIMENSIONS']['options_frame_pady_ratio']))
        self.options_frame.grid_propagate(False)
        self.options_inner = tk.Frame(self.options_frame, bg=CONFIG['COLORS']['options_bg'], 
                                      width=self.left_width - 20, height=self.options_height - 20)
        self.options_inner.grid(row=0, column=0, sticky="nsew")
        self.options_inner.grid_propagate(False)
        self.options_frame.grid_rowconfigure(0, weight=1)
        self.options_frame.grid_columnconfigure(0, weight=1)

        # Right Panel
        if hasattr(self, 'right_panel'):
            self.right_panel.destroy()
        self.right_panel = tk.Frame(self.root, bg=CONFIG['COLORS']['background'], 
                                    width=self.right_width, height=self.screen_height)
        self.right_panel.grid(row=0, column=2, sticky="nsew")
        self.right_panel.grid_propagate(False)
        self.right_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_rowconfigure(2, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        # Timer Label
        if hasattr(self, 'timer_label'):
            self.timer_label.destroy()
        self.timer_label = tk.Label(self.right_panel, text="Time: 00:00", 
                                    bg=CONFIG['COLORS']['background'], fg=CONFIG['COLORS']['timer'])
        self.timer_label.grid(row=0, column=0, sticky="n",
                              padx=int(self.right_width * CONFIG['DIMENSIONS']['timer_padx_ratio']),
                              pady=int(self.screen_height * CONFIG['DIMENSIONS']['timer_pady_ratio']))
        self.timer_label.grid_remove()

        # Submit Button
        if hasattr(self, 'submit_button'):
            self.submit_button.destroy()
        self.submit_button = tk.Button(self.right_panel, text="Submit", command=self.submit, 
                                       fg=CONFIG['COLORS']['submit_fg'], bg=CONFIG['COLORS']['submit_bg'])
        self.submit_button.grid(row=2, column=0, sticky="s",
                                padx=int(self.right_width * CONFIG['DIMENSIONS']['submit_padx_ratio']),
                                pady=int(self.screen_height * CONFIG['DIMENSIONS']['submit_pady_ratio']))
        
        # Vertical Separator
        if hasattr(self, 'separator'):
            self.separator.destroy()
        self.separator = ttk.Separator(self.root, orient='vertical')
        self.separator.grid(row=0, column=1, sticky="ns")

        # Transition Frame
        if hasattr(self, 'transition_frame'):
            self.transition_frame.destroy()
        self.transition_frame = tk.Frame(self.root, bg=CONFIG['COLORS']['background'])

    def _configure_sizes(self):
        """Configure widget sizes and fonts responsively based on current dimensions."""
        self.question_font_size = max(int(((self.left_width+self.screen_height)/2) * CONFIG['FONTS']['question_size_ratio']), 8)
        self.question_text.config(
            font=(CONFIG['FONTS']['family'], self.question_font_size)
        )
        self.option_font_size = max(int(((self.left_width+self.screen_height)/2) * CONFIG['FONTS']['option_size_ratio']), 8)

        self.timer_font_size = max(int(self.right_width * CONFIG['FONTS']['timer_size_ratio']), 8)
        self.timer_label.config(font=(CONFIG['FONTS']['family'], self.timer_font_size))

        submit_width = max(int(self.right_width * CONFIG['DIMENSIONS']['submit_width_ratio']), 10)
        submit_height = max(int(self.screen_height * CONFIG['DIMENSIONS']['submit_height_ratio']), 5)
        
        self.submit_font_size = max(int(((submit_height+submit_width)/2) * CONFIG['FONTS']['submit_size_ratio']), 8)
        
        self.submit_button.config(font=(CONFIG['FONTS']['family'], self.submit_font_size, 'bold'))
        self.submit_button.place(relx=CONFIG['DIMENSIONS']['submit_padx_ratio'], 
                                 rely=CONFIG['DIMENSIONS']['submit_pady_ratio'], 
                                 anchor="center", width=submit_width, height=submit_height)

    def _on_resize(self, event):
        """Handle window resize to maintain responsiveness."""
        new_width = self.root.winfo_width()
        new_height = self.root.winfo_height()
        if new_width != self.screen_width or new_height != self.screen_height:
            self._update_dimensions()
            self._create_panels()
            self._configure_sizes()
            if self.current_index < len(self.question_list):
                if 'separator' in self.question_list[self.current_index]:
                    self._show_separator()
                else:
                    self._display_question(self.question_list[self.current_index])
            elif self.current_index >= len(self.question_list):
                self._show_completion()

    def _display_question(self, question_data):
        """Display a question and its options with fixed sizes."""
        question_info = self.question_bank.get(question_data['question_id'], {})
        self.question_text.config(text=question_info.get('question', 'Question not found'))
        
        options = question_data.get('options', [])
        if not options:
            messagebox.showwarning("Warning", "No options available for this question!")
            return

        for widget in self.options_inner.winfo_children():
            widget.destroy()
        wraplength = int(self.left_width * 0.25)
        columns = CONFIG['DIMENSIONS']['option_button_columns']
        spacing_x = max(int(self.left_width * CONFIG['DIMENSIONS']['option_button_spacing_ratio']), 10)
        spacing_y = max(int(self.options_height * CONFIG['DIMENSIONS']['option_button_margin_ratio']), 10)
        #spacing_x = 100  
        #spacing_y = 50
        fixed_button_height = 6

        for i, opt in enumerate(options):
            text = next((o['text'] for o in question_info.get('options', []) if o['id'] == opt['id']), "Unknown")
            btn = tk.Button(self.options_inner, text=text, wraplength=wraplength, height=fixed_button_height,
                            fg=CONFIG['COLORS']['button_fg'], bg=CONFIG['COLORS']['button_bg'],
                            font=(CONFIG['FONTS']['family'], self.option_font_size),
                            borderwidth=2, relief="solid",
                            command=lambda oid=opt['id'], idx=i: self._select_option(oid, idx))
            btn.grid(row=i // columns, column=i % columns, padx=spacing_x, pady=spacing_y, sticky="nsew")

        rows = (len(options) + columns - 1) // columns
        for col in range(columns):
            self.options_inner.grid_columnconfigure(col, weight=1, uniform="option")
        for row in range(rows):
            self.options_inner.grid_rowconfigure(row, weight=1)

        if self.current_index > self.part1_question_count:
            self.timer_label.grid()
            self.start_timer(CONFIG['TIMING']['part2_duration'])
        else:
            self.timer_label.grid_remove()
            if self.timer_id:
                self.root.after_cancel(self.timer_id)
            if self.blink_id:
                self.root.after_cancel(self.blink_id)
            self.timer_label.config(text="Time: 00:00", fg=CONFIG['COLORS']['timer'])

        csv_path = os.path.join(self.participant_dir, f"Q{self.question_number}.csv")
        self.eye_tracker.start_recording(csv_path)
        self.start_time = time.time()

    def _reset_ui(self):
        """Reset UI components for the next display."""
        self.transition_frame.grid_forget()
        for widget in self.options_inner.winfo_children():
            widget.destroy()
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        self.separator.grid(row=0, column=1, sticky="ns")
        self.right_panel.grid(row=0, column=2, sticky="nsew")

    def _connect_eye_tracker(self):
        """Connect to the eye tracker, handling any errors."""
        # --- Commented out eye tracker connection call ---
        # try:
        #     print("Connecting to eye tracker...")
        #     self.eye_tracker.connect()
        # except Exception as e:
        #     messagebox.showerror("Connection Error", f"Eye tracker connection failed: {e}")
        #     self.root.destroy()
        pass

    def _load_questions(self):
        """Load participant questions and main question bank."""
        try:
            participant_file = os.path.join(CONFIG['PATHS']['question_exams_dir'], f"Participant_{self.participant_id}.json")
            with open(participant_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.question_list = data['Part1'] + [{'separator': True}] + data['Part2']
                self.part1_question_count = len(data['Part1'])
            with open(CONFIG['PATHS']['questions_json'], 'r', encoding='utf-8') as f:
                self.question_bank = {q['id']: q for q in json.load(f)}
            return True
        except Exception as e:
            # Show an error and ensure the main window is closed; return False so caller can stop further initialization
            try:
                messagebox.showerror("Loading Error", f"Failed to load questions: {e}")
            except Exception:
                # messagebox may fail if root is in a bad state
                pass
            try:
                self.root.destroy()
            except Exception:
                pass
            return False

    def display_next(self):
        """Display the next question or transition screen."""
        if self.current_index >= len(self.question_list):
            self._show_completion()
            return
        self._reset_ui()
        if 'separator' in self.question_list[self.current_index]:
            self._show_separator()
        else:
            self._display_question(self.question_list[self.current_index])

    def _show_separator(self):
        """Display a transition screen between Part 1 and Part 2 with enhanced styling."""
        self.transition_frame.grid(row=0, column=0, columnspan=3, sticky="nsew")
    
        # Use a frame with a soft, light blue background
        frame_bg = "#F0F8FF"  # Alice Blue, a very light blue color
        transition_inner = tk.Frame(self.transition_frame, bg=frame_bg, padx=50, pady=40)
        transition_inner.pack(expand=True)
    
        # Sticker element (using an emoji as a sticker)
        sticker_label = tk.Label(
        transition_inner,
        text="🎉",  # Sticker emoji; replace with an image if desired
        font=(CONFIG['FONTS']['family'], self.question_font_size + 10),
        bg=frame_bg
        )
        sticker_label.pack(pady=(0, 20))
    
        # Enhanced transition text with extra line spacing
        transition_text = (
        "Congratulations!\n\n"
        "❤You have successfully completed Part 1❤\n\n"
        "🙏Please get ready to start Part 2🙏\n\n"
        "Each question has a time limit of {} seconds."
        ).format(CONFIG['TIMING']['part2_duration'])
    
        text_label = tk.Label(
        transition_inner,
        text=transition_text,
        font=(CONFIG['FONTS']['family'], self.question_font_size),
        bg=frame_bg,
        fg="#333333",  # A dark gray for elegant contrast
        justify="center"
        )
        text_label.pack(pady=(0, 20))
    
        # Attractive call-to-action button
        start_button = tk.Button(
        transition_inner,
        text="Start Part 2",
        command=self._next_from_separator,
        bg="#28A745",  # Green button background
        fg="#FFFFFF",
        font=(CONFIG['FONTS']['family'], self.option_font_size, 'bold'),
        padx=20,
        pady=10
        )
        start_button.pack()
    
        self.auto_timer = self.root.after(CONFIG['TIMING']['auto_continue'], self._next_from_separator)


    def _next_from_separator(self):
        """Advance from the separator screen to Part 2."""
        if self.auto_timer:
            self.root.after_cancel(self.auto_timer)
        self.current_index += 1
        self.display_next()

    def _select_option(self, option_id, index):
        """Highlight the selected option button."""
        for widget in self.options_inner.winfo_children():
            widget.config(fg=CONFIG['COLORS']['button_fg'], bg=CONFIG['COLORS']['button_bg'])
        self.options_inner.winfo_children()[index].config(fg=CONFIG['COLORS']['selected_fg'], bg=CONFIG['COLORS']['selected_bg'])
        self.selected_option_id = option_id

    def submit(self):
        """Record the selected answer and proceed."""
        if not self.selected_option_id:
            messagebox.showwarning("Warning", "Please select an option!")
            return
        elapsed_time = time.time() - self.start_time
        current_question = self.question_list[self.current_index]
        self.answers.append({
            "section": "Part 2" if self.current_index > self.part1_question_count else "Part 1",
            "question_number": self.question_number,
            "question_id": current_question["question_id"],
            "chosen_option": self.selected_option_id,
            "time_spent": round(elapsed_time, 2)
        })
        self.question_number += 1
        self._advance()

    def start_timer(self, duration):
        """Start a countdown timer for Part 2 questions."""
        self.time_remaining = duration
        self.is_blinking = False
        self._update_timer()

    def _update_timer(self):
        """Update the timer display and handle expiration."""
        mins, secs = divmod(self.time_remaining, 60)
        self.timer_label.config(text=f"Time: {mins:02}:{secs:02}")
        if self.time_remaining <= CONFIG['TIMING']['blink_start'] and not self.is_blinking:
            self.is_blinking = True
            self._blink_timer()
        if self.time_remaining > 0:
            self.time_remaining -= 1
            self.timer_id = self.root.after(1000, self._update_timer)
        else:
            self._handle_timeout()

    def _blink_timer(self):
        if self.is_blinking:
            color = CONFIG['COLORS']['question_bg'] if self.timer_label.cget("fg") == CONFIG['COLORS']['timer'] else CONFIG['COLORS']['timer']
        self.timer_label.config(fg=color)
        #self.question_text.config(bg=color)
        #self.question_frame.config(bg=color)
        #self.question_inner.config(bg=color)
        self.blink_id = self.root.after(CONFIG['TIMING']['blink_interval'], self._blink_timer)


    def _handle_timeout(self):
        """Record a timeout and advance to the next question."""
        current_question = self.question_list[self.current_index]
        self.answers.append({
            "section": "Part 2" if self.current_index > self.part1_question_count else "Part 1",
            "question_number": self.question_number,
            "question_id": current_question["question_id"],
            "chosen_option": None,
            "time_spent": CONFIG['TIMING']['part2_duration']
        })
        self.question_number += 1
        self._advance()

    def _advance(self):
        """Move to the next question after stopping timers and eye tracking."""
        self.eye_tracker.stop_recording()
        # Reset background colors to default
        default_bg = CONFIG['COLORS']['question_bg']
        self.question_text.config(bg=default_bg)
        self.question_frame.config(bg=default_bg)
        self.question_inner.config(bg=default_bg)
        self.timer_label.config(fg=CONFIG['COLORS']['timer'])
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
        if self.blink_id:
            self.root.after_cancel(self.blink_id)
        self.current_index += 1
        self.selected_option_id = None
        self.display_next()


    def _show_completion(self):
        """Display completion message with an exit button and save results."""
        self._reset_ui()
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
        if self.blink_id:
            self.root.after_cancel(self.blink_id)
        if self.auto_timer:
            self.root.after_cancel(self.auto_timer)

        completion_frame = tk.Frame(self.root, bg=CONFIG['COLORS']['background'])
        completion_frame.grid(row=0, column=0, columnspan=3, sticky="nsew")
        tk.Label(completion_frame, text="Exam Completed!\n\n❤Thank you for your cooperation❤", 
                 font=(CONFIG['FONTS']['family'], max(int(self.screen_width * 0.02), 14)), 
                 bg=CONFIG['COLORS']['background']).pack(pady=50)
        exit_button = tk.Button(completion_frame, text="Exit", command=self._exit_application,
                                font=(CONFIG['FONTS']['family'], 12), 
                                bg="#FF4444", fg="#FFFFFF")
        exit_button.place(relx=0.5, rely=0.7, anchor="center", width=500, height=70)
    
        self._save_results()

    def _save_results(self):
        """Save participant answers to a JSON file in participant-specific directory."""
        result_file = os.path.join(self.participant_dir, "answers.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.answers, f, indent=4, ensure_ascii=False)
            
    def _exit_application(self):
        """Clean up resources and exit the application."""
        self.eye_tracker.close()
        self.root.destroy()

# %% [Section 4]: Main Execution
if __name__ == "__main__":
    import tkinter.simpledialog as simpledialog
    
    # Create a temporary main window for the dialog
    temp_root = tk.Tk()
    temp_root.withdraw()
    temp_root.update()  # Update widget states
    
    # Call the dialog without specifying a parent
    PARTICIPANT_ID = simpledialog.askstring(
        "Participant ID",
        "Please enter the participant ID:"
    )
    
    temp_root.destroy()
    
    if PARTICIPANT_ID is None or PARTICIPANT_ID.strip() == "":
        print("No participant ID provided. Application will exit.")
    else:
        root = tk.Tk()
        app = QuestionDisplayApp(root, PARTICIPANT_ID)
        root.mainloop()
        print("Application closed.")
