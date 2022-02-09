"""Houses pinball image utilities in class PinballUtils"""
from curses.panel import bottom_panel, top_panel
import cv2
import numpy as np

from .display_utils import DisplayUtils
from .general_utils import GeneralUtils
from . import config


class PinballUtils:
    def __init__(self, track_pipeline=False):
        self._gen_utils = GeneralUtils()
        self.track_pipeline = track_pipeline
        self._display_pipeline = []

    def get_playfield_corners(frame):
        corner_coordinates = []

        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        # Change to HSV colorspace for color thresholding
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Split frame into two parts to analyze different colors and reduce noise
        [top, bottom] = np.split(hsv, 2, axis=0)
        print(top.shape)    
        print(bottom.shape)

        # Color threshold masks (top of playfield has yellow corner markers and bottom of playfield has blue corner markers)
        yellow_mask = cv2.inRange(top, np.array(config.LOWER_YELLOW), np.array(config.UPPER_YELLOW))
        blue_mask = cv2.inRange(bottom, np.array(config.LOWER_BLUE), np.array(config.UPPER_BLUE))
        # Combining masks to retrieve full frame mask
        mask = np.vstack((yellow_mask, blue_mask))

        # Extract targetted colors from original frame in grayscale
        extracted_colors = cv2.bitwise_and(frame,frame, mask=mask)
        extracted_colors_gray = cv2.cvtColor(extracted_colors, cv2.COLOR_BGR2GRAY)
        extracted_colors_gray = cv2.threshold(extracted_colors_gray, config.CORNERS_BINARY_THRESHOLD_MIN, config.CORNERS_BINARY_THRESHOLD_MAX, cv2.THRESH_BINARY)[1]
        # extracted_colors_gray = cv2.erode(extracted_colors_gray, None, iterations=1)
        # extracted_colors_gray = cv2.dilate(extracted_colors_gray, None, iterations=1)

        # Find contours in the image
        contours, _ = cv2.findContours(extracted_colors_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, config.CONTOUR_APPROXIMATION_COEFFICIENT * perim, True)
            if config.CORNER_CONTOUR_AREA_MIN < area < config.CORNER_CONTOUR_AREA_MAX and len(approx) == 4:
                corner_coordinates.append(c)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
        
        DisplayUtils.display_frame(frame, wait_key=1)

        return corner_coordinates

    def find_corner_rect(
        self, img, user_corners, lower_bound, upper_bound, n_rects, rect_contour_thresh=0
    ):
        img = self._filter_by_proximity(img, user_corners, config.USER_PLAYFIELD_CORNERS_RADIUS)

        img = self._gen_utils.apply_color_filter(img, lower_bound, upper_bound)
        self._append_to_pipeline(img)

        thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
        _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_OTSU)
        self._append_to_pipeline(thresh)

        canny = self._gen_utils.canny(thresh)
        self._append_to_pipeline(canny)

        contours, _ = cv2.findContours(
            canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = self._gen_utils.filter_contours_area(contours, rect_contour_thresh)

        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 3)
        self._append_to_pipeline(contour_img)

        rects = self._gen_utils.get_contours_of_shape(contours, 4)

        rects_img = img.copy()
        cv2.drawContours(rects_img, rects, -1, (255, 0, 0), 3)
        self._append_to_pipeline(rects_img)

        if len(rects) == 0:
            return rects

        rects = self._gen_utils.get_n_largest_contour(rects, n_rects)

        clean_rects = []
        for rect in rects:
            perimeter = cv2.arcLength(rect, True)
            clean_rects.append(cv2.approxPolyDP(rect, 0.05 * perimeter, True))


        # Get just centers of the rects
        centroids = self._gen_utils.get_contour_centers(clean_rects)

        return centroids

    def _filter_by_proximity(self, img, centers_norm, radius_norm):
        # norm radius is a fraction of total size so it needs to be converted to pixel size
        radius = radius_norm * np.sqrt(img.shape[1] * img.shape[0])
        radius = int(radius)
        # TODO this doesn't seem to work
        centers = centers_norm * np.array([img.shape[1], img.shape[0]])
        centers = centers.astype(np.uint64)


        # TODO confirm this works
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for (x, y) in centers:
            mask += self._gen_utils.circular_roi_mask(img, x, y, radius)

        masked = cv2.bitwise_and(img, img, mask=mask)
        return masked

    def _append_to_pipeline(self, img):
        if self.track_pipeline:
            self._display_pipeline.append(img)

    @property
    def display_pipeline(self):
        """Gets the display_pipeline."""
        tmp = self._display_pipeline.copy()
        return tmp

    @display_pipeline.setter
    def display_pipeline(self, new):
        """Sets the display_pipeline."""
        self._display_pipeline = new

    @display_pipeline.deleter
    def display_pipeline(self):
        """Clears the display_pipeline."""
        self._display_pipeline = []

