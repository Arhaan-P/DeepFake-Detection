"""
Logger Utility - Saves terminal output to txt file
===================================================
Automatically logs all print statements to both console and file.
"""

import sys
import os
from datetime import datetime
from pathlib import Path


class TeeLogger:
    """Writes output to both console and file simultaneously."""
    
    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log_dir = Path(log_file).parent
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_file, 'a', encoding='utf-8')
        
        # Write header
        self.log_file.write("\n" + "=" * 70 + "\n")
        self.log_file.write(f"LOG STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 70 + "\n\n")
        self.log_file.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.write("\n" + "=" * 70 + "\n")
        self.log_file.write(f"LOG ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 70 + "\n")
        self.log_file.close()


def setup_logging(script_name: str, log_dir: str = "logs") -> TeeLogger:
    """
    Setup logging to save all terminal output to a txt file.
    
    Args:
        script_name: Name of the script (e.g., 'train', 'preprocess')
        log_dir: Directory to save logs
    
    Returns:
        TeeLogger instance
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.txt")
    
    logger = TeeLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger  # Also capture errors
    
    print(f"Logging to: {log_file}")
    
    return logger


def close_logging(logger: TeeLogger):
    """Close the logger and restore stdout."""
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal
    logger.close()
