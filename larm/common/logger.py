import os
import logging

from larm.common import dist_utils 

ANSI_RESET = "\033[0m"
ANSI_BLUE = "\033[34m"
ANSI_RED = "\033[31m"


class ColorFormatter(logging.Formatter):
    """Formatter that colors the entire log line based on level for console output."""

    LEVEL_TO_COLOR = {
        logging.DEBUG: ANSI_BLUE,
        logging.WARNING: ANSI_RED,
        logging.ERROR: ANSI_RED,
        logging.CRITICAL: ANSI_RED,
    }

    def format(self, record):
        message = super().format(record)
        color = self.LEVEL_TO_COLOR.get(record.levelno)
        if color:
            return f"{color}{message}{ANSI_RESET}"
        return message


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Determine base logging level
    base_level = logging.INFO if dist_utils.is_main_process() else logging.WARN

    # Override with environment variable if provided
    env_level = os.getenv("MEMGEN_LOG_LEVEL")
    if env_level:
        level_name = env_level.strip().upper()
        # Support common aliases
        alias_map = {"WARN": "WARNING", "FATAL": "CRITICAL"}
        level_name = alias_map.get(level_name, level_name)
        if level_name in logging._nameToLevel:
            base_level = logging._nameToLevel[level_name]

    # Configure root logger explicitly (basicConfig with handlers ignores format)
    root_logger = logging.getLogger()
    root_logger.setLevel(base_level)

    # Clear existing handlers to avoid duplicate logs if called multiple times
    root_logger.handlers = []

    # Check if file-only logging is enabled via environment variable
    file_only = os.getenv("LOG_FILE_ONLY", "").lower() in ("1", "true", "yes")

    # Console handler with colors (only if not file-only mode)
    if not file_only:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(base_level)
        stream_formatter = ColorFormatter("%(asctime)s [%(levelname)s] %(message)s")
        stream_handler.setFormatter(stream_formatter)
        root_logger.addHandler(stream_handler)

    # File handler without colors
    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    file_handler.setLevel(base_level)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Log the current mode
    if file_only:
        root_logger.info("Logger initialized in FILE-ONLY mode (console output disabled via LOG_FILE_ONLY)")
    else:
        root_logger.info(f"Logger initialized (output_dir: {output_dir})")