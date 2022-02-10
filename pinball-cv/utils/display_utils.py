"""Houses display image utilities in class DisplayUtils"""
import cv2
import numpy as np


class DisplayUtils:
    # @classmethod
    @staticmethod
    def rotate_frame_counterclockwise(frame):
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    @classmethod
    def resize_img(cls, img, resolution_scale=1):
        return cv2.resize(
            img.copy(),
            (
                int(img.shape[1] * resolution_scale),
                int(img.shape[0] * resolution_scale),
            ),
        )

    @staticmethod
    def display_frame(frame, window_name="test", wait_key=0):
        frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
        cv2.imshow(window_name, frame)
        # cv2.waitKey(wait_key)

    @classmethod
    def detect_draw_contours(cls, src, dest=None):
        """Draws contours from given src on given dst (drawn on src if no dst specified) and returns said contours."""
        contours = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if dest is not None:
                cv2.drawContours(dest, [contour], -1, (255, 0, 0), 3)
            else:
                cv2.drawContours(src, [contour], -1, (255, 0, 0), 3)

        return contours

    @classmethod
    def draw_bboxes(cls, dest, rects, color=(0, 255, 0), thickness=2):
        for rect in rects:
            cv2.rectangle(
                dest, (rect[0], rect[1]), (rect[2], rect[3]), color, thickness
            )

    @classmethod
    def draw_circles(cls, dest, circles, radius=5, color=(0, 255, 0), thickness=-1):
        if isinstance(circles, list) or (
                isinstance(circles, np.ndarray) and len(circles.shape) == 2
        ):
            for circle in circles:
                cv2.circle(dest, (circle[0], circle[1]), radius, color, thickness)
        else:
            for circle in circles:
                cv2.circle(dest, (circle[0][0], circle[0][1]), radius, color, thickness)

    @classmethod
    def create_img_grid(cls, img_matrix):
        grid = []
        for row in img_matrix:
            new_row = []
            for img in row:
                if len(img.shape) != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                new_row.append(img)
            grid.append(np.hstack(new_row))

        return np.vstack(grid)

    @classmethod
    def create_img_grid_list(cls, img_list, width, height):
        assert width * height == len(img_list)
        grid = []
        new_row = []
        for idx, img in enumerate(img_list):
            if img.shape[-1] != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            new_row.append(img)

            if (idx + 1) % width == 0 and idx != 0:
                grid.append(np.hstack(new_row))
                new_row = []

        return np.vstack(grid)
