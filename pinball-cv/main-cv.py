
import numpy as np
import cv2
from torch import frac

from utils import *


# TODOS:
# TODO: add documentation everywhere
# TODO: fix encapsulation by adding _ where applicable


# create utils
gen_utils = GeneralUtils()
# disp_utils = DisplayUtils()
pinball_utils = PinballUtils()
user_input_utils = UserInputUtils()


def main():
    
    cap = cv2.VideoCapture(config.VIDEO_CAPTURE_INPUT)
    print("Video capture initialized")

    runs = 0
    while True:
        ok, curr_frame = cap.read()

        # DisplayUtils.display_frame(curr_frame)

        # Rotate frame
        rotated_frame = DisplayUtils.rotate_frame_counterclockwise(frame=curr_frame)
        # DisplayUtils.display_frame(rotated_frame)
        
        if runs % 20 == 0:
            print("Recalculating Playfield Corners")
            PinballUtils.get_playfield_corners(rotated_frame)
            # playfield_corners = 

    # Release the capture at end
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
