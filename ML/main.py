from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np
import pytesseract
from pytesseract import image_to_string
import matplotlib.pyplot as plt
import PIL
from skimage import exposure

image = cv2.imread("test.jpg")

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

auto = auto_canny(blurred)

# cv2.imshow("Input", edged)
"""cv2.imshow("Input", auto)
cv2.waitKey(0)
exit()"""


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

cv2.imwrite("edited_image.jpg", output)
# output has meter only
#cv2.imshow("Input", output)
#cv2.waitKey(0)



gry_disp_arr = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
gry_disp_arr = exposure.rescale_intensity(gry_disp_arr, out_range= (0,255))
#thresholding
# ret, thresh = cv2.threshold(gry_disp_arr,175,255,cv2.THRESH_BINARY)
ret, thresh = cv2.adaptiveThreshold(gry_disp_arr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY,11,2)

cv2.imshow("",thresh)
cv2.waitKey(0)


exit()


def get_string(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite("removed_noise.png", img)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(PIL.Image.open("removed_noise.png"))

    return result

o = get_string("edited_image.jpg")

print(o)