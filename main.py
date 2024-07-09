import cv2
import numpy as np
from PIL import Image
from doctr.models import ocr_predictor


class ExtractBoxes:
    def __init__(self,
                 box_min_width: int,
                 box_min_height: int,
                 box_max_width: int,
                 box_max_height: int,
                 ):
        self.__box_min_width = box_min_width
        self.__box_min_height = box_min_height
        self.__box_max_width = box_max_width
        self.__box_max_height = box_max_height

    @staticmethod
    def __convert_picture(pil_picture: Image, cv2_picture: np.ndarray) -> np.ndarray:
        if pil_picture is None and cv2_picture is None:
            raise Exception(
                "Either pil_picture or cv2_picture must be provided. pil_picture represents Pillow Image and "
                "cv2_picture represents cv2.Image. You can pass either of these to proceed.")

        if cv2_picture is not None:
            # convert picture to grayscale if in RGB mode
            if len(cv2_picture.shape) > 2:
                return cv2.cvtColor(cv2_picture, cv2.COLOR_BGR2GRAY)
            else:
                print(f"Shape of picture is: {cv2_picture.shape}")
                return cv2_picture

        if pil_picture is not None:
            pic = np.asarray(pil_picture)
            if len(pic.shape) > 2:
                return cv2.cvtColor(pic[:, :, ::-1], cv2.COLOR_BGR2GRAY)
            else:
                return pic

    @staticmethod
    def __sort_contours(contours, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        # construct the list of bounding boxes and sort them from top to
        # bottom
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
                                                 key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return contours, bounding_boxes

    # Functon for extracting the box
    def detect(self, pil_picture: Image = None, cv2_picture: np.ndarray = None):
        self.__picture = self.__convert_picture(pil_picture=pil_picture, cv2_picture=cv2_picture)

        # Binarize and invert image
        (thresh, img_bin) = cv2.threshold(self.__picture, 128, 255,
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
        img_bin = 255 - img_bin  # Invert the image

        # Defining a kernel length
        kernel_length = np.array(self.__picture).shape[1] // 40

        # A vertical kernel of (1 X kernel_length), which will detect all the vertical lines from the image.
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        # A kernel of (3 X 3) ones.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Morphological operation to detect vertical lines from an image
        img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
        vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=3)

        # Morphological operation to detect horizontal lines from an image
        img_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, horizontal_kernel, iterations=3)

        # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
        alpha = 0.5
        beta = 1.0 - alpha

        # This function helps to add two image with
        # specific weight parameter to get a third image as summation of two image.
        img_final_bin = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
        (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours for image, which will detect all the boxes
        contours, hierarchy = cv2.findContours(
            img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort all the contours by top to bottom.
        (contours, boundingBoxes) = self.__sort_contours(contours, method="top-to-bottom")

        for c in contours:
            # Returns the location and width,height for every contour
            x, y, w, h = cv2.boundingRect(c)

            # Only yield cropped picture if the constrains of min and max height succeeds
            is_height_ok = (self.__box_min_height < h < self.__box_max_height)
            is_width_ok = (self.__box_min_width < w < self.__box_max_width)
            if is_width_ok and is_height_ok:
                new_img = self.__picture[y:y + h, x:x + w]
                yield new_img


if __name__ == '__main__':
    # Perform detection using
    picture = Image.open("data/whole.jpg")

    detector = ExtractBoxes(box_min_width=90,
                            box_min_height=50,
                            box_max_width=600,
                            box_max_height=600)
    import time
    model = ocr_predictor('db_resnet50', 'parseq', pretrained=True)
    for i, cropped in enumerate(detector.detect(pil_picture=picture)):
        cropped = Image.fromarray(cropped)
        cropped = cropped.resize((cropped.size[0] * 4, cropped.size[1] * 4))
        cropped = cropped.crop((cropped.width / 2, 0, cropped.width, 140))
        cropped = cropped.convert("RGB")

        start = time.time()
        out = model([np.asarray(cropped)])
        print(f"Model Prediction in {time.time() - start} seconds")
        data = [(word.value, word.confidence) for line in out.pages[0].blocks[0].lines for word in line.words]
        cropped.save(f"pics/{i}__{data[0][0].replace('/', '_')}.png")
        print(data)

        # cropped = cv2.resize(cropped, (cropped.shape[1] * 4, cropped.shape[0] * 4))
        # cv2.imwrite("cropped.png", cropped)
        # cv2.imshow("preview", cropped)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
