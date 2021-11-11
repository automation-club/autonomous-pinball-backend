"""Houses pinball image utilities in class PinballUtils"""
import cv2
import numpy as np

from .general_utils import GeneralUtils
from . import logger
from . import config


class PinballUtils:
    def __init__(self, track_pipeline=False):
        self._gen_utils = GeneralUtils()
        self.track_pipeline = track_pipeline
        self._display_pipeline = []

    def get_playfield_corners(self, img, user_corners):
        # Saving a copy of the image to return after cropping into the field
        org_img = img.copy()
        self._append_to_pipeline(org_img)

        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        self._append_to_pipeline(blurred)

        # for detecting each of the corner tapes
        centroids_yellow = self.find_corner_rect(
            blurred, user_corners, config.LOWER_YELLOW, config.UPPER_YELLOW, 2
        )
        centroids_blue = self.find_corner_rect(
            blurred, user_corners, config.LOWER_BLUE, config.UPPER_BLUE, 2
        )

        if np.array(centroids_yellow).shape != (2, 2) or np.array(
            centroids_blue
        ).shape != (2, 2):
            logger.warning("Unable to the 4 corner points of the playfield!")
            return np.array([])

        centroids_yellow = np.array(centroids_yellow).reshape(2, 2)
        centroids_blue = np.array(centroids_blue).reshape(2, 2)
        centroids = np.append(centroids_yellow, centroids_blue, axis=0)
        return centroids

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

        logger.debug(f"clean rects: {clean_rects}")

        # Get just centers of the rects
        centroids = self._gen_utils.get_contour_centers(clean_rects)
        logger.debug(f"centroids: {centroids}")

        return centroids

    def _filter_by_proximity(self, img, centers_norm, radius_norm):
        # norm radius is a fraction of total size so it needs to be converted to pixel size
        radius = radius_norm * np.sqrt(img.shape[1] * img.shape[0])
        radius = int(radius)
        # TODO this doesn't seem to work
        centers = centers_norm * np.array([img.shape[1], img.shape[0]])
        centers = centers.astype(np.uint64)

        logger.debug(f"Image shape in prox filter: {img.shape}")
        logger.debug(f"Normalized radius: {radius_norm}. Radius: {radius}")
        logger.debug(f"Normalized centers: {centers_norm}. Centers: {centers}")

        # TODO confirm this works
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for (x, y) in centers:
            mask += self._gen_utils.circular_roi_mask(img, x, y, radius)

        masked = cv2.bitwise_and(img, img, mask=mask)
        return masked

    def _append_to_pipeline(self, img):
        if self.track_pipeline:
            self._display_pipeline.append(img)
            logger.debug(
                f"Appended to display_pipeline. New length: {len(self._display_pipeline)}"
            )

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

