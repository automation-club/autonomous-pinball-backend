import numpy as np
import cv2

from img_utils import GeneralUtils, DisplayUtils, PinballUtils
import config

gen_utils = GeneralUtils()
disp_utils = DisplayUtils()
pinball_utils = PinballUtils()

playfield_corners = np.array([])

print("Opening Video Capture")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open video/camera")
    # cap = cv2.VideoCapture(str(config.VIDEO_PATH))

print("Starting")

# keep track of how many frames have been seen
frame_count = 0
while True:
    # Use a timer to keep track of speed of algorithm
    timer = cv2.getTickCount()

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_count += 1
    if not ret:
        break

    # Apply algorithm
    if playfield_corners.size != 0:
        frame = gen_utils.unwarp_rect(frame, playfield_corners)

    # Only find playfield once every n frames to save compute
    if frame_count % config.UPDATE_PLAYFIELD_INTERVAL == 0:
        playfield = pinball_utils.get_playfield_corners(frame)
        if playfield.size != 0:
            playfield_corners = playfield
        else:
            playfield_corners = np.array([])

    # Calculate FPS before displaying
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    fps = round(fps, 2)

    ### Displaying ###
    # Selecting which frame to display
    disp_img = frame

    if config.CONSOLE_OUTPUT:
        # print("Res: ", frame.shape)
        print("playfield corners", playfield_corners)

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
            disp_utils.resize_img(disp_img, resolution_scale=config.RESOLUTION_SCALE),
        )

    if cv2.waitKey(1) & 0xFF == 27:
        break


# Release the capture at end
cap.release()
cv2.destroyAllWindows()
