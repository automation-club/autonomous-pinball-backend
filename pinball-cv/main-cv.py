"""Runs the main loop to detect the pinball on the playfield."""
import logging

import numpy as np
import cv2

from utils import *
import config
from config import logger  # so that we don't need to say config.logger every time


# create utils
gen_utils = GeneralUtils()
disp_utils = DisplayUtils()
pinball_utils = PinballUtils()

# houses corners of playfield for unwarpping
playfield_corners = np.array([])

logger.info(f"Opening video capture camera {config.VIDEO_CAPTURE_NUMBER}")
cap = cv2.VideoCapture(config.VIDEO_CAPTURE_NUMBER)
if not cap.isOpened():
    logger.error(f"Cannot open camera {config.VIDEO_CAPTURE_NUMBER}")

    logger.info("Opening video capture camera 0")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error(f"Cannot open camera 0")

        logger.info(f"Opening video at {config.VIDEO_PATH}")
        cap = cv2.VideoCapture(str(config.VIDEO_PATH))
        if not cap.isOpened():
            logger.critical(f"Cannot open video at {config.VIDEO_PATH}")
            exit()

logger.info("==============================\nStarting ball tracking\n==============================")


frame_count = 0
"""int: number of frames since start of program."""

while True:
    # Use a timer to keep track of speed of algorithm
    timer = cv2.getTickCount()

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_count += 1
    if not ret:
        logger.error("Failed to read from video capture")
        continue

    # Apply algorithm

    # If there is no playfield detected, find it
    # Also re-find playfield once every n frames to save compute and account for slight movements
    if playfield_corners.size == 0 or frame_count % config.UPDATE_PLAYFIELD_INTERVAL == 0:
        playfield = pinball_utils.get_playfield_corners(frame)
        logger.INFO(f"Detected playfield corners: {playfield}")
        if playfield.size != 0:
            playfield_corners = playfield
        else:
            playfield_corners = np.array([])

    # If there is already a playfield detected, unwarp the frame with saved corners
    if playfield_corners.size != 0:
        frame = gen_utils.unwarp_rect(frame, playfield_corners)

    # Displaying

    # Calculate FPS to know the algorithm efficiency
    # Done before displaying as we are only interested in the efficiency of the algorithm itself, not the displaying
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    fps = round(fps, 2)

    # Displaying on a copied frame
    disp_img = frame.copy()

    if config.DRAW_OUTPUT:
        cv2.putText(
            disp_img,
            f"FPS {int(fps)}",
            (0, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 170, 50),
            2,
        )

        cv2.imshow(
            "video feed",
            disp_utils.resize_img(disp_img, resolution_scale=config.DISPLAY_RESOLUTION_SCALE),
        )

    # Stop the program if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break


# Release the capture at end
cap.release()
cv2.destroyAllWindows()
