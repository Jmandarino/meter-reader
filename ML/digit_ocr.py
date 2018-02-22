import numpy as np
import cv2
import imutils
from skimage import exposure
from pytesseract import image_to_string
import PIL
from imutils.perspective import four_point_transform
import os


def take_picture(should_save=False, d_id=0):
    cam = cv2.VideoCapture(d_id)
    s, img = cam.read()
    if s:
        if should_save:
            cv2.imwrite('ocr.jpg',img)
        print( "picture taken")
    return img

def cnvt_edged_image(img_arr, should_save=False):
    # ratio = img_arr.shape[0] / 300.0
    image = imutils.resize(img_arr,height=300)
    gray_image = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),11, 17, 17)
    edged_image = cv2.Canny(gray_image, 30, 200)

    if should_save:
        cv2.imwrite('cntr_ocr.jpg')

    return edged_image

'''image passed in must be ran through the cnv_edge_image first'''
def find_display_contour(edge_img_arr):
    display_contour = None
    edge_copy = edge_img_arr.copy()
    contours,hierarchy = cv2.findContours(edge_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    top_cntrs = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    for cntr in top_cntrs:
        peri = cv2.arcLength(cntr,True)
        approx = cv2.approxPolyDP(cntr, 0.02 * peri, True)

        if len(approx) == 4:
            display_contour = approx
            break

    return display_contour

def crop_display(image_arr):
    edge_image = cnvt_edged_image(image_arr)
    display_contour = find_display_contour(edge_image)
    cntr_pts = display_contour.reshape(4,2)
    return cntr_pts


def normalize_contrs(img,cntr_pts):
    ratio = img.shape[0] / 300.0
    norm_pts = np.zeros((4,2), dtype="float32")

    s = cntr_pts.sum(axis=1)
    norm_pts[0] = cntr_pts[np.argmin(s)]
    norm_pts[2] = cntr_pts[np.argmax(s)]

    d = np.diff(cntr_pts,axis=1)
    norm_pts[1] = cntr_pts[np.argmin(d)]
    norm_pts[3] = cntr_pts[np.argmax(d)]

    norm_pts *= ratio

    (top_left, top_right, bottom_right, bottom_left) = norm_pts

    width1 = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    width2 = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    height1 = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    height2 = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    max_width = max(int(width1), int(width2))
    max_height = max(int(height1), int(height2))

    dst = np.array([[0,0], [max_width -1, 0],[max_width -1, max_height -1],[0, max_height-1]], dtype="float32")
    persp_matrix = cv2.getPerspectiveTransform(norm_pts,dst)
    return cv2.warpPerspective(img,persp_matrix,(max_width,max_height))

def process_image(orig_image_arr):
    ratio = orig_image_arr.shape[0] / 300.0

    display_image_arr = normalize_contrs(orig_image_arr,crop_display(orig_image_arr))
    #display image is now segmented.
    gry_disp_arr = cv2.cvtColor(display_image_arr, cv2.COLOR_BGR2GRAY)
    gry_disp_arr = exposure.rescale_intensity(gry_disp_arr, out_range= (0,255))

    #thresholding
    ret, thresh = cv2.threshold(gry_disp_arr,127,255,cv2.THRESH_BINARY)
    return thresh

def ocr_image(orig_image_arr):
    otsu_thresh_image = PIL.Image.fromarray(process_image(orig_image_arr))
    return image_to_string(otsu_thresh_image, lang="letsgodigital", config="-psm 100 -c tessedit_char_whitelist=.0123456789")



def process_image_2(path_to_img):
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

    display, output = process_image_2('test.jpg')

    thresh = cv2.threshold(display, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    thresh_copy = thresh.copy()
    thresh_copy = cv2.dilate(thresh_copy, kernel, iterations=1)
    _, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #_, cnts, hierarchy = cv2.findContours(thresh_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    digitCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 3 and h > 10:
            digitCnts.append(c)

    cnt = digitCnts[0]

    # dig_flat = digitCnts.flatten()
    # cnt = dig_flat[0]

    # hull = cv2.convexHull(cnt,returnPoints = False)
    # defects = cv2.convexityDefects(cnt,hull)
    #
    # for i in range(defects.shape[0]):
    #     s,e,f,d = defects[i,0]
    #     start = tuple(cnt[s][0])
    #     end = tuple(cnt[e][0])
    #     far = tuple(cnt[f][0])
    #     cv2.line(output,start,end,[0,255,0],2)
    #     cv2.circle(output,far,5,[0,0,255],-1)

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

    thresh_crop = thresh[min_y -5 :max_y+5, min_x-5:max_x+5]
    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(output, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
    # for cnt in digitCnts:
    #     # get convex hull
    #     hull = cv2.convexHull(cnt)
    #     # draw it in red color
    #     cv2.drawContours(output, [hull], -1, (0, 0, 255), 1)

    # cv2.drawContours(output, digitCnts, -1, (0, 255, 0), 1)
    # cv2.imshow("output without drawcontours()", output)
    # cv2.waitKey(0)


    #
    # np.set_printoptions(threshold=np.nan)
    # out = open('whites.txt','w')
    # out.write(str(whites))
    # out.close()
    # TODO: get path to this file
    os.environ["TESSDATA_PREFIX"] = "/Users/joey/PycharmProjects/meter-reader/ML/tessdata"


    # cv2.drawContours(output, digitCnts, -1, (0, 255, 0), 1)
    # cv2.imshow("output without drawcontours()", thresh_crop)
    # cv2.waitKey(0)
    otsu_thresh_image = thresh_crop #PIL.Image.fromarray(thresh)

    out = image_to_string(otsu_thresh_image, lang="letsgodigital", config="-psm 8 -c tessedit_char_whitelist=0123456789")

    print("printing::::")
    print(out)
    exit(0)

