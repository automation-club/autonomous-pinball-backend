"""Configures and stores constants for running the autonomous pinball backend."""
from pathlib import Path
import logging


# Logging
LOG_PATH = "./logs/cv.log"
"""Path: path to log file."""
LOG_TO_FILE = True
"""bool: whether to log to a log file."""

logger = logging.getLogger(__name__)
"""Logger: logging object for pinball-cv."""
logger.setLevel(logging.INFO)

# Sets formatting for loggers
file_format = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
stream_format = logging.Formatter("%(name)s:%(levelname)s:%(message)s")

# Creates handlers
file_handler = logging.FileHandler(LOG_PATH)
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(file_format)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(stream_format)

if LOG_TO_FILE:
    logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Functional
UPDATE_PLAYFIELD_INTERVAL = 60
"""int: number of frames per update of corners of playfield."""
VIDEO_CAPTURE_NUMBER = 1
"""int: the capture camera number to try to read from."""

# Datasets
DATASET_PATH = Path("./datasets/")
"""Path: path to any datasets."""
VIDEO_PATH = DATASET_PATH / "playing-phone-trim.mp4"
"""Path: path to any test videos."""
IMG_PATH = DATASET_PATH / "pinball-tape-img.jpg"
"""Path: path to any test images."""


# Displaying
DRAW_OUTPUT = True
"""bool: whether to display the live feed on screen."""
DISPLAY_RESOLUTION_SCALE = 1
"""int: the resolution to display the live feed at."""
