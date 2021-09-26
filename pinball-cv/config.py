from pathlib import Path

CONSOLE_OUTPUT = True
DRAW_OUTPUT = True
RESOLUTION_SCALE = 1

DATASET_PATH = Path("./datasets/")
VIDEO_PATH = DATASET_PATH / "playing-phone-trim.mp4"
IMG_PATH = DATASET_PATH / "pinball-tape-img.jpg"

UPDATE_PLAYFIELD_INTERVAL = 60
"""int: number of frames per update of corners of playfield"""
