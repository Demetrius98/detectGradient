class xyPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rectangle:
    def __init__(self, point_1, point_2):
        self.point_1 = point_1
        self.point_2 = point_2

    def height(self):
        return abs(self.point_2.y - self.point_1.y)

    def width(self):
        return abs(self.point_2.x - self.point_1.x)

    def size(self):
        return self.width() * self.height()