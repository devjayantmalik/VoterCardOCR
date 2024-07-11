import os.path
import pymupdf
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import sqlite3
import time
import concurrent.futures


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

            # Only yield cropped picture if the constraints of min and max height succeeds
            is_height_ok = (self.__box_min_height < h < self.__box_max_height)
            is_width_ok = (self.__box_min_width < w < self.__box_max_width)
            if is_width_ok and is_height_ok:
                new_img = self.__picture[y:y + h, x:x + w]
                yield new_img


def create_db():
    conn = sqlite3.connect('data/database.db')
    cursor = conn.cursor()

    # Create the devices table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS devices (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        batch_size INTEGER DEFAULT 50
    )
    ''')

    # Create the jobs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY,
        device_id INTEGER,
        pdf_filename TEXT,
        picture_base64 TEXT,
        prediction TEXT,
        prediction_export TEXT,
        confidence REAL,
        is_task_complete BOOLEAN DEFAULT FALSE,
        errors TEXT DEFAULT NULL,
        uid TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (device_id) REFERENCES devices (id)
    )
    ''')

    # Commit the changes and close the connection
    conn.commit()
    return conn, cursor


def read_doc_pages(path: str) -> Image:
    doc = pymupdf.open(path)  # open a document
    for page_index in range(len(doc)):  # iterate over pdf pages
        page = doc[page_index]  # get the page
        image_list = page.get_images()

        page = image_list[0][0]
        pix = pymupdf.Pixmap(doc, page)
        if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

        img_data = pix.tobytes("png")  # get the image data as bytes
        image = Image.open(BytesIO(img_data))  # open it as a PIL image
        yield image


def clean_and_preprocess(box: np.ndarray):
    cropped = Image.fromarray(box)
    cropped = cropped.resize((cropped.size[0] * 4, cropped.size[1] * 4))
    cropped = cropped.crop((cropped.width / 2, 0, cropped.width, 140))
    cropped = cropped.convert("RGB")
    buffered = BytesIO()
    cropped.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())


def process_pdf(filepath: str):
    try:
        pst = time.time()
        data = []
        filename = os.path.basename(filepath)
        pages = list(read_doc_pages(filepath))

        for page_no, page in enumerate(pages):
            if page_no == 0 or page_no == 1 or page_no == len(pages) - 1:
                continue

            card_nos = [clean_and_preprocess(pic) for pic in detector.detect(pil_picture=page)]

            # prepare data for insertion in db
            data += [(filename, card, f'{filename}_{page_no}_{idx}') for idx, card in enumerate(card_nos)]
        
        global processed_files
        global total_files
        processed_files += 1
        print(f"Processed ({processed_files}/{total_files}) :{filepath} in {time.time() - pst} seconds")
        # Return query and associated data
        data_insert_query = '''INSERT OR IGNORE INTO jobs (pdf_filename,picture_base64, uid) VALUES (?, ?, ?);'''
        return data_insert_query, data
    except Exception as ex:
        print(f"Error {ex}")
        return None, None

total_files = 0
processed_files = 0
detector = ExtractBoxes(box_min_width=295,
                        box_min_height=120,
                        box_max_width=600,
                        box_max_height=600)
if __name__ == '__main__':
    # create database
    conn, cursor = create_db()

    start = time.time()
    files_path = os.path.join("data", "documents")

    files = [os.path.join(files_path, filename) for filename in os.listdir(files_path)]
    total_files = len(files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(process_pdf, files)
        for q, data in results:
            conn.executemany(q, data)
            conn.commit()
    print(f"All pdfs processed in {time.time() - start} seconds")
