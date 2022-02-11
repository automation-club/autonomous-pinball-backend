"""Houses pinball image utilities in class PinballUtils"""
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

    @staticmethod
    def get_playfield_corners(frame):
        """Identifies coordinates of the playfield corners from the frame.

        Parameters
        ----------
        frame : numpy.ndarray
            The frame read in by the VideoCapture

        Returns
        -------
        corners_coordinates : list
            List of coordinates of the identified playfield corners
        """

        corner_coordinates = []

        blurred_frame = cv2.GaussianBlur(frame, config.GAUSSIAN_BLUR_KERNEL_SIZE, 0)
        # Change to HSV colorspace for color thresholding
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        # Split frame into two parts to analyze different colors and reduce noise
        [top, bottom] = np.split(hsv, 2, axis=0)

        # Color threshold masks (top of playfield has yellow corner markers and bottom of playfield has blue corner
        # markers)
        yellow_mask = cv2.inRange(top, np.array(config.CORNER_LOWER_YELLOW), np.array(config.CORNER_UPPER_YELLOW))
        blue_mask = cv2.inRange(bottom, np.array(config.CORNER_LOWER_BLUE), np.array(config.CORNER_UPPER_BLUE))
        # Combining masks to retrieve full frame mask
        mask = np.vstack((yellow_mask, blue_mask))

        # Extract targeted colors from original frame in grayscale
        extracted_colors = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask)
        extracted_colors_gray = cv2.cvtColor(extracted_colors, cv2.COLOR_BGR2GRAY)

        # Find contours in the image
        contours, _ = cv2.findContours(extracted_colors_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, config.CONTOUR_APPROXIMATION_COEFFICIENT * perim, True)
            if config.CORNER_CONTOUR_AREA_MIN < area < config.CORNER_CONTOUR_AREA_MAX and len(approx) == 4:
                m = cv2.moments(c)
                x = int(m["m10"] / m["m00"])
                y = int(m["m01"] / m["m00"])
                corner_coordinates.append([x, y])

        return corner_coordinates

    @staticmethod
    def warp_frame(frame, playfield_corners):
        """
        Warps the frame to the playfield.

        Parameters
        ----------
        frame : np.ndarray
            The frame to transform.
        playfield_corners : list
            The coordinates of the corners of the playfield.

        Returns
        -------
        np.ndarray
            The warped frame.
        """
        playfield_corners_sorted = PinballUtils.sort_coordinates(playfield_corners)
        display_corners_sorted = PinballUtils.sort_coordinates([
            [0, 0],
            [frame.shape[1], 0],
            [frame.shape[1], frame.shape[0]],
            [0, frame.shape[0]]
        ])

        # Calculate transformation matrix
        transformation_matrix = cv2.getPerspectiveTransform(playfield_corners_sorted, display_corners_sorted)

        return cv2.warpPerspective(frame, transformation_matrix, (frame.shape[1], frame.shape[0]))

    @staticmethod
    def sort_coordinates(coordinates):
        """
        Sorts the coordinates of the playfield corners.

        Parameters
        ----------
        coordinates : list
            The coordinates of the corners of the playfield.

        Returns
        -------
        np.ndarray
            The coordinates of the corners of the playfield, sorted.
        """

        coords = np.array(coordinates, dtype=np.float32)
        sorted_coords = np.zeros(coords.shape, dtype=np.float32)

        s = np.sum(coords, axis=1)
        sorted_coords[0] = coords[np.argmax(s)]  # Top left corner
        sorted_coords[2] = coords[np.argmin(s)]  # Bottom right corner
        diff = np.diff(coords, axis=1)
        sorted_coords[1] = coords[np.argmin(diff)]  # Top right corner
        sorted_coords[3] = coords[np.argmax(diff)]  # Bottom left corner

        return sorted_coords

    @staticmethod
    def get_pinball_coordinates(frame, prev_frame):
        """
        Identifies the coordinates of the pinball in the frame.

        Parameters
        ----------
        frame : np.ndarray
            The frame to analyze.

        Returns
        -------
        pinball_coordinates : list
            The coordinates of the pinball.
        """

        pinball_coordinates = []

        # Blur frame, convert to HSV and threshold to isolate pinball color
        blurred_frame = cv2.GaussianBlur(frame, config.GAUSSIAN_BLUR_KERNEL_SIZE, 0)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        extracted_color_binary = cv2.inRange(hsv, np.array(config.PINBALL_LOWER_COLOR),
                                             np.array(config.PINBALL_UPPER_COLOR))

        # Find contours from the extracted color image
        contours = cv2.findContours(extracted_color_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find all contours that fit parameters
        for c in contours[0]:
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, config.CONTOUR_APPROXIMATION_COEFFICIENT*perim, True)
            if config.PINBALL_CONTOUR_MIN_AREA < area < config.PINBALL_CONTOUR_MAX_AREA \
                    and perim < config.PINBALL_CONTOUR_MAX_PERIMETER \
                    and len(approx) < config.PINBALL_CONTOUR_MAX_SIDES:
                m = cv2.moments(c)
                x = int(m['m10'] / m['m00'])
                y = int(m['m01'] / m['m00'])
                pinball_coordinates.append([x, y])

        return pinball_coordinates

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
