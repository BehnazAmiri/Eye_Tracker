"""Logging utilities."""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """Simple logger that writes to console and file."""
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize logger.
        
        Args:
            log_file: Path to log file (optional)
        """
        self.log_file = log_file
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str, also_print: bool = True):
        """
        Log message.
        
        Args:
            message: Message to log
            also_print: Whether to also print to console
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        if also_print:
            print(log_message)
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
    
    def section(self, title: str):
        """Log a section header."""
        separator = "=" * 80
        self.log(separator)
        self.log(title)
        self.log(separator)


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger.
    """
    def __init__(self, logger: Logger):
        self.logger = logger
    
    def write(self, message: str):
        if message.strip():  # Avoid logging empty lines
            self.logger.log(message.rstrip(), also_print=False)
    
    def flush(self):
        pass
