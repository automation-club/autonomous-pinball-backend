"""Runs the main loop to detect the pinball on the playfield."""

import numpy as np
import cv2

# import utils.config as config  # so that we don't need to say utils.config every time
# from utils.config import logger  # so that we don't need to say config.logger every time
# from utils.general_utils import GeneralUtils
# from utils.display_utils import DisplayUtils
# from utils.pinball_utils import PinballUtils
from utils import *


# TODOS:
# TODO: add documentation everywhere
# TODO: fix ndarray type hinting everywhere.
# TODO: fix encapsulation by adding _ where applicable


# create utils
gen_utils = GeneralUtils()
disp_utils = DisplayUtils()
pinball_utils = PinballUtils(track_pipeline=config.DISPLAY_PIPELINE)
user_input_utils = UserInputUtils()


def main():
    # houses corners of playfield for unwarpping
    playfield_corners = np.array([])
    user_playfield_corners = np.array([])

    logger.info(f"Opening video capture {config.VIDEO_CAPTURE_INPUT}")
    cap = cv2.VideoCapture(config.VIDEO_CAPTURE_INPUT)
    if not cap.isOpened():
        logger.critical(f"Cannot open video capture {config.VIDEO_CAPTURE_INPUT}")
        exit()

    logger.info(
        "\n==============================\nStarting ball tracking\n=============================="
    )

    frame_count = 0
    """int: number of frames since start of program."""

    frame = np.array([])
    while True:
        # Use a timer to keep track of speed of algorithm
        timer = cv2.getTickCount()

        # Capture frame-by-frame
        prev_frame = frame
        ret, frame = cap.read()
        frame_count += 1
        if not ret:  # if we fail to read in the frame (likely because we reached end of video)
            logger.exception("Failed to read from video capture")
            frame = prev_frame
            # break

        # if user has not yet selected playfield corners
        if user_playfield_corners.size == 0:
            user_playfield_corners = user_input_utils.get_user_clicks(frame, 4, "Select 4 playfield corners.")
            logger.info(f"User selected corners (as fractions): {user_playfield_corners}")

        # Apply algorithm

        # If there is no playfield detected, find it
        # Also re-find playfield once every n frames to save compute and account for slight movements
        if (
                playfield_corners.size == 0
                or frame_count % config.UPDATE_PLAYFIELD_INTERVAL == 0
        ):
            # Clear the display pipeline so we can see a fresh pipeline
            del pinball_utils.display_pipeline
            playfield = pinball_utils.get_playfield_corners(frame)
            logger.info(f"Detected playfield corners: {playfield}")
            if playfield.size != 0:
                playfield_corners = playfield
            else:
                playfield_corners = np.array([])

        # If there is already a playfield detected, unwarp the frame with saved corners
        if playfield_corners.size != 0:
            frame_unwarp = gen_utils.warp_img_to_rect(frame, playfield_corners)
        else:
            frame_unwarp = frame.copy()

        # Displaying

        # Calculate FPS to know the algorithm efficiency
        # Done before displaying as we are only interested in the efficiency of the algorithm itself, not the displaying
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        fps = round(fps, 2)

        # Display a copy of the image which we draw on
        disp_img = frame.copy()
        # disp_img = frame_unwarp.copy()

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

            disp_utils.draw_circles(disp_img, playfield_corners, radius=10)
            cv2.imshow(
                "video feed",
                disp_utils.resize_img(
                    disp_img, resolution_scale=config.DISPLAY_RESOLUTION_SCALE
                ),
            )

            if config.DISPLAY_PIPELINE:
                pipeline = pinball_utils.display_pipeline
                cv2.imshow(
                    "video pipeline",
                    DisplayUtils.resize_img(
                        DisplayUtils.create_img_grid_list(
                            pipeline,
                            len(pipeline) // 2,
                            len(pipeline) // (len(pipeline) // 2),
                        ),
                        resolution_scale=0.25,
                    ),
                )

        # Stop the program if ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the capture at end
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
