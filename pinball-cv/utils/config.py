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
USER_PLAYFIELD_CORNERS_RADIUS = 0.1
"""float: the radius around the user selected playfield corners to search in for the playfield corners, 
as a fraction of the entire frame. """
# Colored tapes bounds config
# Which bounds to look in for each color in HSV format
LOWER_YELLOW = [20, 235, 100]
UPPER_YELLOW = [30, 255, 200]
LOWER_BLUE = [0, 0, 0]
UPPER_BLUE = [255, 255, 255]

# Displaying
DRAW_OUTPUT = True
"""bool: whether to display the live feed on screen."""
DISPLAY_RESOLUTION_SCALE = 1
"""int: the resolution to display the live feed at."""
DISPLAY_PIPELINE = True
"""bool: whether to display the image pipeline on the screen."""
