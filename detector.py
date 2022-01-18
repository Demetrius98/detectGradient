from PIL import Image
import numpy as np
from rectangleSearch import RectangleSearch


class RectangleGradientDetector:
    # White color
    COLOR_CODE_BEFORE_DETECT = 1
    # Black color
    COLOR_CODE_AFTER_DETECT = 0

    def __init__(self, image, colors_limit=3, size_limit=25, length_limit=5, gradient_delta=2.5):
        self._image = np.asarray(image, dtype='float32')
        self._colors_limit = colors_limit
        self._size_limit = size_limit
        self._length_limit = length_limit
        self._gradient_delta = gradient_delta

    def detect(self) -> Image:
        rectangle_search = RectangleSearch(self._image, self._colors_limit, self._size_limit, self._length_limit,
                                           self._gradient_delta)
        rectangles = rectangle_search.get_results()

        res_img = np.full(self._image.shape[:2], self.COLOR_CODE_BEFORE_DETECT)
        for rectangle in rectangles:
            for i in range(rectangle.point_1.x, rectangle.point_2.x):
                for j in range(rectangle.point_1.y, rectangle.point_2.y):
                    res_img[j][i] = self.COLOR_CODE_AFTER_DETECT

        img_res = Image.fromarray(res_img.astype('uint8') * 255)
        return img_res
