import time
import cv2

from utils import *

# TODO: add documentation everywhere


def main():
    cap = cv2.VideoCapture(config.VIDEO_CAPTURE_INPUT)
    print("Video capture initialized")

    prev_time = 0
    new_time = 0
    fond = cv2.FONT_HERSHEY_DUPLEX
    playfield_corners = None
    prev_frame = None
    runs = 0
    while True:
        # Read frame
        ok, curr_frame = cap.read()
        if not ok:
            print("Error reading frame")
            break

        # Rotate frame
        rotated_frame = DisplayUtils.rotate_frame_counterclockwise(frame=curr_frame)

        # Recalculate playfield corner coordinates every 30 frames
        if runs % 30 == 0 or playfield_corners is None:
            playfield_corners_temp = PinballUtils.get_playfield_corners(rotated_frame)
            if len(playfield_corners_temp) == 4:
                print(f"Playfield corners found - Run: {runs}")
                playfield_corners = playfield_corners_temp

        # Warp frame to playfield
        if playfield_corners is not None:
            warped_frame = PinballUtils.warp_frame(rotated_frame, playfield_corners)
        else:
            warped_frame = rotated_frame

        # Pinball detection
        if runs == 0:
            print(f"Looking for {config.PINBALL_COLOR} pinball")
        pinball_coordinates = PinballUtils.get_pinball_coordinates(warped_frame, prev_frame)

        # Draw pinball coordinates
        if len(pinball_coordinates) == 1:
            DisplayUtils.draw_circles(warped_frame, pinball_coordinates, radius=8)

        new_time = time.time()
        fps = f"FPS: {int(1 / (new_time - prev_time))}"
        cv2.putText(warped_frame, fps, (10, 30), fond, 1, (255, 255, 255), 2, cv2.LINE_AA)
        prev_time = new_time
        prev_frame = rotated_frame

        # Display frame
        DisplayUtils.display_frame(warped_frame, "Post-processed frame")

        # Break out of loop if Q key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stop program key pressed")
            break
        runs += 1

    # Release the capture at end of program
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
