import numpy as np
import cv2
import imutils
import PIL
from imutils.perspective import four_point_transform
import os
from imutils import contours

def process_image(path_to_img):
    # creates an edge map
    image = cv2.imread(path_to_img)
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
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
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    output = four_point_transform(image, displayCnt.reshape(4, 2))

    # output is the display contoured in gray scale
    return warped, output


def trim(im):
    bg = PIL.Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = PIL.ImageChops.difference(im, bg)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


if __name__ == '__main__':

    display, output = process_image('img/test.jpg')

    # cv2.imshow("output without drawcontours()", output)
    # cv2.waitKey(0)

    thresh = cv2.threshold(display, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # cv2.imshow("output without drawcontours()", thresh)
    # cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


    # cv2.imshow("output without drawcontours()", thresh)
    # cv2.waitKey(0)

    thresh_copy = thresh.copy()
    thresh_copy = cv2.dilate(thresh_copy, kernel, iterations=2)
    # thresh_copy = cv2.erode(thresh_copy, kernel, iterations=2)
    _, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #_, cnts, hierarchy = cv2.findContours(thresh_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    digitCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 4 and h > 12:
            digitCnts.append(c)

    cnt = digitCnts[0]

    try: hierarchy = hierarchy[0]
    except: hierarchy = []

    height, width, _ = output.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(digitCnts, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        # if w > 80 and h > 80:
        #     cv2.rectangle(output, (x,y), (x+w,y+h), (255, 0, 0), 2)

    thresh_crop = thresh[min_y:max_y, min_x:max_x]
    out_crop = output[min_y:max_y , min_x:max_x]
    # cv2.imshow("output without drawcontours()", out_crop)
    # cv2.waitKey(0)



    kernel = np.ones((1,1),np.uint8)
    # thresh_crop = cv2.erode(thresh_crop, kernel, iterations=2)
    # thresh_crop = cv2.dilate(thresh_crop, kernel, iterations=2)
    # thresh_crop = cv2.morphologyEx(thresh_crop, cv2.MORPH_CLOSE, kernel)


    # refind contours after croping image
    gray = cv2.cvtColor(out_crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # remove noise
    kernel = np.ones((1,1),np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    kernel = np.ones((1,1),np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=5)

    # make them longer
    kernel = np.ones((3,1),np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)


    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # cv2.drawContours(thresh, digitCnts, -1, (0,255,0), 1)
    # cv2.imshow("output without drawcontours()", thresh)
    # cv2.waitKey(0)

    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    _, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    digitCnts = []
    boxes = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 4 and h > 12:
            digitCnts.append(c)

    # process the image 2 times, first time gets a close contour
    # TODO: get path to this file
    os.environ["TESSDATA_PREFIX"] = "/Users/joey/PycharmProjects/meter-reader/ML/tessdata"

    # process individual characters
    output_list = []
    digitCnts = contours.sort_contours(digitCnts,
                                 method="left-to-right")[0]
    BLACK = [0,0,0]
    for counter, c in enumerate(digitCnts[::-1]):
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]
        constant = cv2.copyMakeBorder(roi,5,5,10,10,cv2.BORDER_CONSTANT,value=BLACK)
        cv2.imwrite("test-"+str(counter)+".bmp", constant)



    exit(0)

