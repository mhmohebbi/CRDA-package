"""
Configurable logging utilities for CRDA.

This module provides the Logger class, a singleton logger that supports
output to console, file, or both. It wraps Python's standard logging
module with a simpler interface.

Example:
    Basic console logging::

        from crda.utils.logger import Logger

        logger = Logger(log_to_console=True)
        logger.info("Starting experiment...")
        logger.warning("Low sample count detected")

    File and console logging::

        logger = Logger(
            log_to_console=True,
            log_to_file=True,
            log_file="experiment.log"
        )
        logger.info("Results will be saved to experiment.log")
"""

import logging
import os
import sys
from typing import Optional, Union


class Logger:
    """Configurable singleton logger for console and file output.

    Implements a singleton pattern to ensure consistent logging throughout
    the application. Supports customizable output destinations, log levels,
    and formatting.

    Args:
        name: Logger name for identification. Defaults to "cfda".
        log_to_console: Enable console output. Defaults to True.
        log_to_file: Enable file output. Defaults to False.
        log_file: Path to log file. If None and log_to_file is True,
            defaults to 'logs/{name}.log'.
        log_level: Logging level. Can be int (e.g., logging.INFO) or string
            (e.g., "INFO", "DEBUG"). Defaults to logging.INFO.

    Attributes:
        name: The logger's name.
        logger: The underlying logging.Logger instance.

    Note:
        Due to the singleton pattern, only the first instantiation's
        parameters take effect. Subsequent instantiations return the
        same logger instance.

    Example:
        >>> logger = Logger(name="crda", log_to_console=True)
        >>> logger.info("Experiment started")
        2026-01-12 14:30:52 - INFO - Experiment started
        >>> logger.set_level("DEBUG")
        >>> logger.debug("Detailed debugging info")
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton logger instance."""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        name: str = "cfda",
        log_to_console: bool = True,
        log_to_file: bool = False,
        log_file: Optional[str] = None,
        log_level: Union[int, str] = logging.INFO
    ):
        """Initialize the logger with specified configuration.

        Args:
            name: Logger name for identification.
            log_to_console: Whether to output logs to console (stdout).
            log_to_file: Whether to output logs to a file.
            log_file: Log file path. If None and log_to_file is True,
                defaults to 'logs/{name}.log'.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                Can be int or string.
        """
        if self._initialized:
            return

        self.name = name
        self.logger = logging.getLogger(name)

        # Convert string log level to int if needed
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper())

        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear any existing handlers

        # Create formatters
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

        # Add file handler if requested
        if log_to_file:
            if log_file is None:
                # Create default logs directory
                os.makedirs('logs', exist_ok=True)
                log_file = f'logs/{name}.log'

            # Create directory for log file if needed
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)

        self._initialized = True

    def debug(self, message: str) -> None:
        """Log a debug-level message.

        Args:
            message: The message to log.
        """
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log an info-level message.

        Args:
            message: The message to log.
        """
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning-level message.

        Args:
            message: The message to log.
        """
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error-level message.

        Args:
            message: The message to log.
        """
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log a critical-level message.

        Args:
            message: The message to log.
        """
        self.logger.critical(message)

    def set_level(self, level: Union[int, str]) -> None:
        """Set the logging level.

        Args:
            level: New logging level. Can be int (e.g., logging.DEBUG)
                or string (e.g., "DEBUG").
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.setLevel(level)

    def add_file_handler(self, log_file: str) -> None:
        """Add a file handler to output logs to a file.

        Args:
            log_file: Path to the log file.
        """
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Get the underlying Python logger object.

        Returns:
            The logging.Logger instance.
        """
        return self.logger
