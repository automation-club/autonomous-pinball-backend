"""Configures and stores constants for running the autonomous pinball backend."""
from pathlib import Path

# Datasets
# Resolve changes symlinks to absolute path
DATASET_DIRECTORY_PATH = Path("../datasets/").resolve()
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



# Playfield Corner Detection Tunable Paramaeters
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
"""Tuple[int, int]: kernel size of the Gaussian Blur."""
CONTOUR_APPROXIMATION_COEFFICIENT = 0.1
"""float: coefficient of epsilon in polygon approximation of the contour."""
CORNERS_BINARY_THRESHOLD_MIN = 20
"""int: minimum threshold for the binary thresholding of the playfield corners."""
CORNERS_BINARY_THRESHOLD_MAX = 255
"""int: maximum threshold for the binary thresholding of the playfield corners."""
CORNER_CONTOUR_AREA_MIN = 30
"""int: minimum area of the contour of the playfield corners."""
CORNER_CONTOUR_AREA_MAX = 250
"""int: maximum area of the contour of the playfield corners."""

# HSV color mapping intervals to find pinball playfield corners
LOWER_YELLOW = [20, 130, 130]
UPPER_YELLOW = [30, 255, 255]
LOWER_BLUE = [100, 140, 20]
UPPER_BLUE = [130, 255, 255]


