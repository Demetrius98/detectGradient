from PIL import Image
from detector import RectangleGradientDetector
import sys

DEFAULT_OPEN_IMAGE_PATH = 'test.png'
DEFAULT_RESULT_IMAGE_PATH = 'result.png'

if __name__ == '__main__':
    if len(sys.argv) == 2:
        image_open_name = sys.argv[1]
        image_result_name = DEFAULT_RESULT_IMAGE_PATH
    elif len(sys.argv) == 3:
        image_open_name = sys.argv[1]
        image_result_name = sys.argv[2]
    else:
        image_open_name = DEFAULT_OPEN_IMAGE_PATH
        image_result_name = DEFAULT_RESULT_IMAGE_PATH

    image = Image.open(image_open_name)
    rectangle_gradient_detector = RectangleGradientDetector(image)
    image_res = rectangle_gradient_detector.detect()
    image_res.save(image_result_name)
    print("Image has been processed!")