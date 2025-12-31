"""Timing utilities for performance monitoring."""

import time
from typing import Optional
from datetime import datetime


class TimedLogger:
    """Logger that adds elapsed time to each message."""

    def __init__(self, prefix: str = ""):
        """
        Initialize timed logger.

        Args:
            prefix: Optional prefix for all log messages
        """
        self.start_time = time.time()
        self.prefix = prefix

    def reset(self):
        """Reset the timer."""
        self.start_time = time.time()

    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    def log(self, message: str, reset: bool = False):
        """
        Print message with elapsed time.

        Args:
            message: Message to print
            reset: If True, reset timer after logging
        """
        elapsed = self.elapsed()
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix_str = f"[{self.prefix}] " if self.prefix else ""
        print(f"[{timestamp}] [{elapsed:6.2f}s] {prefix_str}{message}")

        if reset:
            self.reset()

    def section(self, title: str, width: int = 60):
        """
        Print a section header with timing.

        Args:
            title: Section title
            width: Width of separator line
        """
        elapsed = self.elapsed()
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*width}")
        print(f"[{timestamp}] [{elapsed:6.2f}s] {title}")
        print(f"{'='*width}")


def timed_print(message: str, start_time: Optional[float] = None):
    """
    Print message with timestamp and optional elapsed time.

    Args:
        message: Message to print
        start_time: Optional start time for elapsed calculation
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    if start_time is not None:
        elapsed = time.time() - start_time
        print(f"[{timestamp}] [{elapsed:6.2f}s] {message}")
    else:
        print(f"[{timestamp}] {message}")
