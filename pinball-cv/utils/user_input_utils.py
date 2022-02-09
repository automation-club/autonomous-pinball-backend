import cv2
import numpy as np

from numpy.typing import NDArray



class UserInputUtils:
    def __init__(self):
        self._user_clicks = []
        """list: a list of the locations user has clicked.
        
        Format: list of length 2 lists containing x, y coordinate pairs, where x and y are non-normalized pixel 
        coordinates."""

    def _listener_user_clicks(self, event, x, y, flags, param):
        # this function follows opencv's specifications for event listeners.
        if event == cv2.EVENT_LBUTTONDOWN:
            self._user_clicks.append([x, y])

    def get_user_clicks(
        self, img: NDArray, num_points: int = 1, window_name: str = "Make selection",
    ) -> NDArray:
        """Gets the points the user clicked on the provided image.

        Parameters
        ----------
        img : NDArray
            The image to get clicks on.
        num_points : int
            The number of clicks to get before returning.
        window_name : str
            The name of the window that will display the img.

        Returns
        -------
        points : NDArray
            A numpy array of points clicked on by the user of shape `(num_points, 2)`, where each row is a point clicked
            on by the user, with x and y values normalized to `img` size.
        """
        self._clear_user_clicks()  # clear old corners first

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._listener_user_clicks)

        while len(self._user_clicks) < num_points:  # show image until we get 4 corners
            cv2.imshow(window_name, img)

            if cv2.waitKey(1) & 0xFF == 27:
                # TODO fix this case where exiting before all n points
                break

        cv2.destroyWindow(window_name)
        # normalize corners to image size
        # first number in image shape is the number of rows, second is the number of columns
        width = img.shape[1]
        height = img.shape[0]

        clicks = np.array(self._user_clicks)
        return clicks / [
            width,
            height,
        ]  # divides first column by width, second by height

    def _clear_user_clicks(self) -> None:
        """Clears internal _user_clicks list.

        Returns
        -------
        None
        """
        self._user_clicks = []
