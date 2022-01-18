import numpy as np
from rectangle import xyPoint, Rectangle


class RectangleSearch:
    DEFAULT_EPS = 1e-7

    def __init__(self, image, colors_limit=3, size_limit=100, length_limit=5, gradient_delta=0.6):
        self._image = image
        self._colors_limit = colors_limit
        self._size_limit = size_limit
        self._length_limit = length_limit
        self._gradient_delta = gradient_delta

        self._y_gradients, self._y_back_gradients = self.find_line_gradients(self._image)
        self._x_gradients, self._x_back_gradients = self.find_line_gradients(np.transpose(self._image, (1, 0, 2)))

        self._last_positions = np.full(self._image.shape[0], 0)

    def is_gradient_numbers(self, numbers) -> bool:
        n = len(numbers)
        x = np.arange(n)
        y = np.array(numbers)

        x_sum = x.sum()
        y_sum = y.sum()

        a = (n * (x * y).sum() - x_sum * y_sum) / (n * (x ** 2).sum() - x_sum ** 2)
        b = (y_sum - a * x_sum) / n

        for x_i, y_i in zip(x, y):
            y_pred = a * x_i + b
            if y_pred < y_i - self._gradient_delta - self.DEFAULT_EPS or y_pred > y_i + self._gradient_delta + self.DEFAULT_EPS:
                return False

        return True

    def is_gradient_pixels(self, line_pixels) -> bool:
        for numbers in zip(*line_pixels):
            if not self.is_gradient_numbers(numbers):
                return False
        return True

    def find_line_gradients(self, image_pixels) -> tuple:
        result = np.full(image_pixels.shape[:2], 0)
        result_back = np.full(image_pixels.shape[:2], 0)

        for i, line_pixels in enumerate(image_pixels):
            j = 0
            while j < len(line_pixels) - 1:
                lower = j + 2
                upper = len(line_pixels) + 1

                while upper - lower > 1:
                    middle = lower + int((upper - lower + 1) / 2)
                    if self.is_gradient_pixels(line_pixels[j:middle]):
                        lower = middle
                    else:
                        upper = middle

                for k in range(j, lower):
                    result[i][k] = lower
                    result_back[i][k] = j
                j = lower - 1

        return result, result_back

    def check_region(self, region) -> bool:
        if region.size() < self._size_limit:
            return False
        if region.height() < self._length_limit:
            return False
        if region.width() < self._length_limit:
            return False

        colors = set()
        for x in range(region.point_1.x, region.point_2.x):
            for y in range(region.point_1.y, region.point_2.y):
                colors.add(';'.join(map(str, self._image[y][x])))
                if len(colors) >= self._colors_limit:
                    return True
        return False

    def extend_position(self, start, x_level, y_lower, y_upper) -> tuple:
        while y_lower >= 0 and \
                self._y_gradients[y_lower][start.x] >= x_level and \
                self._x_gradients[x_level][y_lower] == self._x_gradients[x_level][start.y]:
            self._last_positions[y_lower] = max(self._last_positions[y_lower], x_level)
            y_lower -= 1
        y_lower += 1

        while y_upper < self._image.shape[0] and \
                self._y_gradients[y_upper][start.x] >= x_level and \
                self._x_gradients[x_level][y_upper - 1] == self._x_gradients[x_level][start.y]:
            self._last_positions[y_upper] = max(self._last_positions[y_upper], x_level)
            y_upper += 1

        return y_lower, y_upper

    def next_regions(self):
        y_start = np.argmin(self._last_positions)
        x_start = self._y_back_gradients[y_start][self._last_positions[y_start]]
        start = xyPoint(x_start, y_start)

        x_level = self._y_gradients[y_start][start.x] - 1
        y_lower, y_upper = self.extend_position(start, x_level, y_start, y_start + 1)
        yield Rectangle(xyPoint(x_start, y_lower), xyPoint(x_level + 1, y_upper))

        while start.x <= x_level:
            x_level = start.x - 1

            if y_lower > 0:
                x_level_new = self._y_gradients[y_lower - 1][start.x] - 1
                if x_level_new < x_level and \
                        self._x_gradients[x_level_new][y_lower - 1] == self._x_gradients[x_level_new][start.y]:
                    x_level = max(x_level, x_level_new)

            if y_upper < self._image.shape[0] - 1:
                x_level_new = self._y_gradients[y_upper][start.x] - 1
                if x_level_new < x_level and \
                        self._x_gradients[x_level_new][y_upper] == self._x_gradients[x_level_new][start.y]:
                    x_level = max(x_level, x_level_new)

            y_lower, y_upper = self.extend_position(start, x_level, y_lower, y_upper)
            yield Rectangle(xyPoint(x_start, y_lower), xyPoint(x_level + 1, y_upper))

    def get_results(self) -> list:
        res_list = list()
        while np.min(self._last_positions) < self._image.shape[1] - 1:
            for region in self.next_regions():
                if self.check_region(region):
                    res_list.append(region)

        return res_list
