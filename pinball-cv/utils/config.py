"""Configures and stores constants for running the autonomous pinball backend."""
from pathlib import Path

# Datasets
# Resolve changes symlinks to absolute path
DATASET_DIRECTORY_PATH = Path("./datasets/").resolve()
"""Path: path to any datasets."""
VIDEO_PATH: Path = (DATASET_DIRECTORY_PATH / "green.mp4").resolve()
"""Path: path to any test videos."""

# Functional
UPDATE_PLAYFIELD_INTERVAL = 60
"""int: number of frames per update of corners of playfield."""
VIDEO_CAPTURE_NUMBER = None
"""int: the capture camera number to try to read from."""
VIDEO_CAPTURE_INPUT = str(VIDEO_PATH)
"""Any: the actual input to the cv2.VideoCapture function. Can be swapped between VIDEO_CAPTURE_NUMBER, VIDEO_PATH, 
IMG_PATH, etc. """
# HSV color mapping intervals to find pinball playfield corners
LOWER_YELLOW = [20, 235, 100]
UPPER_YELLOW = [30, 255, 200]
LOWER_BLUE = [0, 0, 0]
UPPER_BLUE = [255, 255, 255]

