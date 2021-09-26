import cv2
import numpy as np


class PinballUtils:
    def __init__(self):
        self._gen_utils = GeneralUtils()
        self.display_pipeline = []

    def get_playfield_corners(self, img):
        # Saving a copy of the image to return after cropping into the field
        org_img = img.copy()
        self.display_pipeline.append(org_img)

        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        self.display_pipeline.append(blurred)

        lower_yellow = [15, 100, 200]
        upper_yellow = [40, 255, 255]
        lower_blue = [100, 200, 200]
        upper_blue = [200, 255, 255]

        # for detecting each of the corner tapes
        centroids_yellow = self.find_corner_rect(blurred, lower_yellow, upper_yellow, 2)
        centroids_blue = self.find_corner_rect(blurred, lower_blue, upper_blue, 2)

        if np.array(centroids_yellow).shape != (2, 2) or np.array(
            centroids_blue
        ).shape != (2, 2):
            print("Unable to the 4 corner points of the playfield!")
            return np.array([])

        centroids_yellow = np.array(centroids_yellow).reshape(2, 2)
        centroids_blue = np.array(centroids_blue).reshape(2, 2)
        centroids = np.append(centroids_yellow, centroids_blue, axis=0)
        return centroids


    def get_playfield(self, img):
        centroids = self.get_playfield_corners()

        rects_img = img.copy()
        DisplayUtils.draw_circles(rects_img, centroids)
        self.display_pipeline.append(rects_img)

        unwarped = self._gen_utils.unwarp_rect(img, centroids)
        self.display_pipeline.append(unwarped)
        return unwarped



    def find_corner_rect(
        self, img, lower_bound, upper_bound, n_rects, rect_contour_thresh=0
    ):
        img = self._gen_utils.apply_color_filter(img, lower_bound, upper_bound)
        self.display_pipeline.append(img)

        thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
        _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_OTSU)
        self.display_pipeline.append(thresh)

        canny = self._gen_utils.canny(thresh)
        self.display_pipeline.append(canny)

        contours, _ = cv2.findContours(
            canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = self._gen_utils.filter_contours_area(contours, rect_contour_thresh)

        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 3)
        self.display_pipeline.append(contour_img)

        rects = self._gen_utils.get_contours_of_shape(contours, 4)

        rects_img = img.copy()
        cv2.drawContours(rects_img, rects, -1, (255, 0, 0), 3)
        self.display_pipeline.append(rects_img)

        if len(rects) == 0:
            return rects

        rects = self._gen_utils.get_n_largest_contour(rects, n_rects)

        clean_rects = []
        for rect in rects:
            perimeter = cv2.arcLength(rect, True)
            clean_rects.append(cv2.approxPolyDP(rect, 0.05 * perimeter, True))

        print("clean rects:", clean_rects)

        # Get just centers of the rects
        centroids = self._gen_utils.get_contour_centers(clean_rects)
        print("centroids:", centroids)

        return centroids


class GeneralUtils:
    @classmethod
    def canny(cls, img, lower_thresh=None, upper_thresh=None, clean=True):
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
    def sort_rect_points(cls, points_new):
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
    def unwarp_rect(cls, img, rect_points):
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
            print(h)
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


class DisplayUtils:
    @classmethod
    def resize_img(cls, img, resolution_scale=1):
        return cv2.resize(
            img.copy(),
            (
                int(img.shape[1] * resolution_scale),
                int(img.shape[0] * resolution_scale),
            ),
        )

    @classmethod
    def display_img(cls, img, window_name="img", wait_key=-1, resolution_scale=1):
        cv2.imshow(window_name, cls.resize_img(img, resolution_scale))
        cv2.waitKey(wait_key)

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


if __name__ == "__main__":
    pinball_util = PinballUtils()
    img = cv2.imread("./datasets/pinball-tape-img-redo.jpg")
    playfield = pinball_util.get_playfield_corners(img)
    cv2.imshow("playfield", playfield)

    DisplayUtils.display_img(
        DisplayUtils.create_img_grid_list(
            pinball_util.display_pipeline,
            len(pinball_util.display_pipeline) // 2,
            len(pinball_util.display_pipeline)
            // (len(pinball_util.display_pipeline) // 2),
        ),
        resolution_scale=0.25,
    )

    cv2.waitKey(-1)
