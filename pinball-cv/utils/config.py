"""Configures and stores constants for running the autonomous pinball backend."""
from pathlib import Path
import logging


# Logging
FILE_LOG_LEVEL = logging.WARNING
STREAM_LOG_LEVEL = logging.DEBUG
LOG_TO_FILE = False
"""bool: whether to log to a log file."""
LOG_PATH = Path("../logs/cv.log")
"""Path: path to log file."""

# Datasets
# resolve changes symlinks to absolute path
DATASET_PATH = Path("./datasets/").resolve()
"""Path: path to any datasets."""
VIDEO_PATH: Path = (DATASET_PATH / "corner-detection.mp4").resolve()
"""Path: path to any test videos."""
IMG_PATH = DATASET_PATH / "pinball-tape-img-redo.jpg"
"""Path: path to any test images."""

# Functional
UPDATE_PLAYFIELD_INTERVAL = 60
"""int: number of frames per update of corners of playfield."""
VIDEO_CAPTURE_NUMBER = 2
"""int: the capture camera number to try to read from."""
VIDEO_CAPTURE_INPUT = str(VIDEO_PATH)
"""Any: the actual input to the cv2.VideoCapture function. Can be swapped between VIDEO_CAPTURE_NUMBER, VIDEO_PATH, 
IMG_PATH, etc. """
# Colored tapes bounds config
# Which bounds to look in for each color in HSV format
LOWER_YELLOW = [15, 100, 200]
UPPER_YELLOW = [40, 255, 255]
LOWER_BLUE = [100, 200, 200]
UPPER_BLUE = [200, 255, 255]

# Displaying
DRAW_OUTPUT = True
"""bool: whether to display the live feed on screen."""
DISPLAY_RESOLUTION_SCALE = 1
"""int: the resolution to display the live feed at."""
DISPLAY_PIPELINE = True
"""bool: whether to display the image pipeline on the screen."""