"""Houses general image utilities in class GeneralUtils"""

import cv2
import numpy as np


class GeneralUtils:
    """Houses general utilities for manipulating images with OpenCV and Numpy."""
    @classmethod
    def canny(cls, img, lower_thresh=None, upper_thresh=None, clean=True):
        # Applies canny edge detection on given image.
        # The image to apply canny edge detection on.
        # The image to apply canny edge detection on.
        if lower_thresh is None or upper_thresh is None:
            median = np.median(img)
            sigma = 0.33
            lower_thresh = (
                int(max(0, (1.0 - sigma) * median))
                if lower_thresh is None
                else lower_thresh
            )
            upper_thresh = (
                int(min(255, (1.0 + sigma) * median))
                if upper_thresh is None
                else upper_thresh
            )
        canny = cv2.Canny(img, lower_thresh, upper_thresh)

        if clean:
            kernel = np.ones((5, 5), np.uint8)
            canny = cv2.dilate(canny, kernel, iterations=2)
            canny = cv2.erode(canny, kernel, iterations=1)

        return canny

    @classmethod
    def apply_color_filter(cls, img, lower_bound, upper_bound):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        res = cv2.bitwise_and(img, img, mask=mask)

        # slow, will skip for now
        # cleaned = cv2.morphologyEx(res, cv2.MORPH_OPEN, (5, 5), iterations=1)
        return cv2.GaussianBlur(res, (3, 3), 0)

    @classmethod
    def sort_rect_points(cls, points_new: np.ndarray) -> np.ndarray:
        """Sorts provided points of a rectangle.

        Follows format: TODO

        Parameters
        ----------
        points_new : np.ndarray
            The points of the rectangle to sort.

        Returns
        -------

        """
        points_old = np.array(points_new).reshape(4, 2)
        points_new = np.zeros((4, 1, 2), dtype=np.int32)
        sum = points_old.sum(1)

        points_new[0] = points_old[np.argmin(sum)]
        points_new[3] = points_old[np.argmax(sum)]
        diff = np.diff(points_old, axis=1)
        points_new[1] = points_old[np.argmin(diff)]
        points_new[2] = points_old[np.argmax(diff)]
        return points_new

    @classmethod
    def warp_img_to_rect(cls, img: np.ndarray, rect_points: np.ndarray) -> np.ndarray:
        """Warps given image to the given rectangle points array.

        Parameters
        ----------
        img : np.ndarray
            The image to warp.
        rect_points : np.ndarray
            The points of the new corners of the image to warp to. Must be valid points on the provided image.
            Follows format TODO.

        Returns
        -------
        new_img : np.ndarray
            The warped image.
        """

        rect_points = cls.sort_rect_points(rect_points).astype(np.float32)
        corner_points = np.array(
            [
                [0, 0],
                [img.shape[1], 0],
                [0, img.shape[0]],
                [img.shape[1], img.shape[0]],
            ],
            dtype=np.float32,
        )

        trans_mat = cv2.getPerspectiveTransform(rect_points, corner_points)
        return cv2.warpPerspective(img, trans_mat, (img.shape[1], img.shape[0]))


    @classmethod
    def circular_roi_mask(cls, img, x, y, radius):
        height, width = img.shape[:2]
        # Confirm x,y coordinate is inside image
        assert 0 <= x < width
        assert 0 <= y < height

        # first get rectangular roi contain the circle
        # x, y is in the center of the rectangle (square)
        # x1 = x - radius
        # x2 = x + radius
        # y1 = y - radius
        # y2 = y + radius
        # rect = img[y1:y2, x1:x2, :]

        # create a mask of same size as rect
        mask = np.zeros(img.shape[:2], dtype=img.dtype)
        cv2.circle(mask, (x, y), radius, 255, -1)

        return mask


    @classmethod
    def get_contour_centers(cls, contours):
        centeriods = []
        for contour in contours:
            M = cv2.moments(contour)
            center_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            center_y = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
            centeriods.append(np.array([center_x, center_y]))
        return centeriods

    @classmethod
    def get_bbox_centers(cls, bboxes):
        centeriods = []
        for bbox in bboxes:
            centeriods.append(
                np.array([(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2])
            )
        return centeriods

    @classmethod
    def calculate_iou(cls, bbox1, bbox2):
        """Calculates IOU of given two bounding boxes in form of numpy array [x1, y1, x2, y2]"""
        assert bbox1[0] <= bbox1[2]
        assert bbox1[1] <= bbox1[3]
        assert bbox2[0] <= bbox2[2]
        assert bbox2[1] <= bbox2[3]

        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])

        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
        bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    @classmethod
    def generate_bboxes(cls, contours):
        bounding_rects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_rects.append(np.array([x, y, x + w, y + h]))

        return bounding_rects

    @classmethod
    def get_distinct_bboxes(cls, bboxes, threshold):
        distinct_bboxes = []
        for i, bbox1 in enumerate(bboxes):
            distinct = True
            for j in range(i + 1, len(bboxes)):
                bbox2 = bboxes[j]
                if cls.calculate_iou(bbox1, bbox2) > threshold:
                    distinct = False
                    break

            if distinct:
                distinct_bboxes.append(bbox1)

        return distinct_bboxes

    @classmethod
    def connect_similar_contours(cls, contours):
        rects = cls.generate_bboxes(contours)
        new_rects, weights = cv2.groupRectangles(rects, 1)
        return new_rects, weights

    @classmethod
    def get_n_largest_contour(cls, contours, n=1):
        contours.sort(reverse=True, key=lambda i: cv2.contourArea(i))
        return contours[:n]

    @classmethod
    def filter_contours_area(cls, contours, area_thresh, keep_big=True):
        filtered = []
        for contour in contours:
            if keep_big and cv2.contourArea(contour) > area_thresh:
                filtered.append(contour)
            elif not keep_big and cv2.contourArea(contour) < area_thresh:
                filtered.append(contour)

        return filtered

    @classmethod
    def filter_contours_closed(cls, contours, hierarchy):
        # TODO need to fix to make this work
        filtered = []
        for contour, h in zip(contours, hierarchy):
            if cv2.isContourConvex(contour) and h[2] != -1:
                filtered.append(contour)

        return filtered

    @classmethod
    def blacken_small_noise(cls, img, contours, area_thresh):
        # TODO finish
        contours = cls.filter_contours_area(contours, area_thresh, keep_big=False)
        for contour in contours:
            if cv2.isContourConvex(contour):
                cv2.fillPoly(img, contour, (0, 0, 0))
                # cv2.fillConvexPoly(img, contour, (0, 0, 0))

        return img

    @classmethod
    def equalize_hist_colored(cls, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img

    @classmethod
    def unsharp_mask(cls, img, blur_size, img_weight, gaussian_weight):
        # code from https://stackoverflow.com/questions/42872353/correcting-rough-edges/42872732
        gaussian = cv2.GaussianBlur(img, blur_size, 0)
        return cv2.addWeighted(img, img_weight, gaussian, gaussian_weight, 0)

    @classmethod
    def smoother_edges(
        cls,
        img,
        first_blur_size,
        second_blur_size=(5, 5),
        img_weight=1.5,
        gaussian_weight=-0.5,
    ):
        # code from https://stackoverflow.com/questions/42872353/correcting-rough-edges/42872732
        # blur the image before unsharp masking
        img = cv2.GaussianBlur(img, first_blur_size, 0)
        # perform unsharp masking
        return cls.unsharp_mask(img, second_blur_size, img_weight, gaussian_weight)

    @classmethod
    def detect_shape(cls, contour, num_sides):
        """Returns True if shape with provided number of sides is detected with supplied closed contour and is
        convex, False otherwise. """
        # must be closed contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        if len(approx) == num_sides and cv2.isContourConvex(approx):
            return True

        return False

    @classmethod
    def get_contours_of_shape(cls, contours, num_sides):
        new_contours = []
        for contour in contours:
            if cls.detect_shape(contour, num_sides):
                new_contours.append(contour)

        return new_contours


def main():
    """Run simple tests to see if the processing of the image using these utils works."""
    general_utils = GeneralUtils()
    img = cv2.imread("./datasets/pinball-tape-img-redo.jpg")

    mask = general_utils.circular_roi_mask(img, img.shape[1] // 2, img.shape[0] // 2, 150)

    new_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("img", img)
    cv2.imshow("mask", mask)
    cv2.imshow("new", new_img)

    cv2.waitKey(-1)


if __name__ == "__main__":  # only runs when this isn't used as a module
    main()
