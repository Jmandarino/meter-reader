import os
import numpy as np
import cv2
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import json


def sort_contours(cnts, method="right-to-left"):
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
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def process_image(path_to_img):
    # creates an edge map and convert to gray scale
    image = cv2.imread(path_to_img)
    image = imutils.resize(image, height=1500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    # find contours in the edge map, then sort them by their
    # size in descending order
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if the contour has four vertices, then we have found
        # the thermostat display
        if len(approx) == 4:
            displayCnt = approx
            break
    # find the display aka the first 4 sided object we see
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    output = four_point_transform(image, displayCnt.reshape(4, 2))

    # output is the display contoured in gray scale
    return warped, output


def thresh_and_crop(display, output):
    """Threshold image, find countours, and crop the display and output image

    :param display: grayscale image cropped to meter LCD screen
    :param output: Original image cropped to meter LCD screen
    :return:
    """

    thresh = cv2.threshold(display, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh_copy = thresh.copy()
    thresh_copy = cv2.dilate(thresh_copy, kernel, iterations=2)
    # thresh_copy = cv2.erode(thresh_copy, kernel, iterations=2)
    _, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > h:
            continue
        aspect = float(w) /h
        # remove contours with bad aspect ratios
        if aspect <.2:
            continue

        if w > 4 and h > 12:
            digitCnts.append(c)

    height, width, _ = output.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    # computes the bounding box for the contour, and draws it on the frame,
    for contour in digitCnts:
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        # if w > 80 and h > 80:
        #     cv2.rectangle(output, (x,y), (x+w,y+h), (255, 0, 0), 2)

    # crop the threshold copy and the B&W output based on where the numbers are
    thresh_crop = thresh[min_y:max_y, min_x:max_x]
    out_crop = output[min_y:max_y , min_x:max_x]

    return thresh_crop, out_crop


def get_digits(out_crop):
    """

    :param out_crop: an image of the tightly cropped meter values
    :return: list of digits contours
    """
    # adjust exposure
    alpha = float(2.5)
    exposure_img = cv2.multiply(out_crop, np.array([alpha]))
    gray = cv2.cvtColor(exposure_img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # remove noise
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=5)

    # then stretch the image vertically to connect the segments
    kernel = np.ones((3, 1), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    _, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # processing again and resetting variables
    digitCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        aspect = float(w) /h
        # remove contours with bad aspect ratios
        if aspect <.2:
            continue
        # contours that are more wide than tall aren't numbers
        if w > h:
            continue
        # if the contour is too small we can assume it isn't a digit and noise
        if w > 4 and h > 12:
            digitCnts.append(c)

    return thresh, digitCnts


if __name__ == '__main__':
    """Process the image twice, the first time we find the bounding area of the contours of
    the numbers
    
    the next time processing we extract the numbers and process
    
    """

    # BASE_PATH = os.path.realpath(__file__)
    DIR_NAME = os.path.dirname(__file__)
    IN_DIR = "in"
    IMG_DIR = "imgs"
    JSON_FILE = "images.json"
    DIGITS = "digits"
    DIGIT_PATH = os.path.abspath(os.path.join(DIR_NAME, IN_DIR, IMG_DIR, DIGITS))
    JSON_PATH = os.path.abspath(os.path.join(DIR_NAME, IN_DIR,JSON_FILE))
    image_list = []

    try:
        data = json.load(open(JSON_PATH, 'r'))
    except FileExistsError:
        print("couldn't find JSON")
        exit(1)

    # store path to image in image list from json file
    for image in data["body"]["images"]:
        t = (image["name"], image["imagePath"], image["folderPath"] )
        image_list.append(t)

    for image_tuple in image_list:
        name = image_tuple[0]
        DIGIT_DIR = image_tuple[2]
        if not os.path.exists(DIGIT_DIR):
            os.makedirs(DIGIT_DIR)
        # process image and obtain display
        display, output = process_image(image_tuple[1])

        # save display for debugging/testing purposes
        cv2.imwrite(os.path.join(DIGIT_DIR,  "screen.bmp"), display)



    # crop image down to just bigger than the size of the digits
        thresh_crop, out_crop = thresh_and_crop(display, output)

        # processing the image 2 times ended up being more successful
        # refind contours after croping image
        thresh, digitCnts = get_digits(out_crop)

        # process individual characters
        digitCnts = contours.sort_contours(digitCnts, method="right-to-left")[0] # this might not be working

        BLACK = [0, 0, 0] # define color for border
        for counter, c in enumerate(digitCnts[::-1]):
            # extract the digit ROI
            (x, y, w, h) = cv2.boundingRect(c)
            # cut image down to size of contour's max boundaries
            roi = thresh[y:y + h, x:x + w]
            # add a border, this has helped with image processing via KNN/SVM
            constant = cv2.copyMakeBorder(roi, 5, 5, 10, 10, cv2.BORDER_CONSTANT, value=BLACK)
            path = os.path.join(DIGIT_DIR,  str(counter) + ".bmp")
            cv2.imwrite(os.path.join(DIGIT_DIR,  str(counter) + ".bmp"), constant)
            cv2.imwrite(os.path.join(DIGIT_DIR,  "thresh" + ".bmp"), thresh)
            # TODO: make this its own method