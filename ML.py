# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:38:53 2015

@author: Pavitrakumar
Edited by: Jmandarino

"""

import numpy as np
from scipy.misc.pilutil import imresize
import cv2 #version 3.2.0
from skimage.feature import hog
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

DIGIT_WIDTH = 10
DIGIT_HEIGHT = 20
IMG_HEIGHT = 28
IMG_WIDTH = 28
CLASS_N = 10 # 0-9

def split2d(img, cell_size, flatten=True):
    """
    This method splits the input training image into small cells (of a single digit) and uses these cells as training data.
    The default training image (MNIST) is a 1000x1000 size image and each digit is of size 10x20. so we divide 1000/10 horizontally and 1000/20 vertically.

    :param img:
    :param cell_size:
    :param flatten:
    :return:
    """
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells


def load_digits(fn):
    print('loading "%s for training" ...' % fn)
    digits_img = cv2.imread(fn, 0)
    digits = split2d(digits_img, (DIGIT_WIDTH, DIGIT_HEIGHT))
    resized_digits = []
    for digit in digits:
        resized_digits.append(imresize(digit,(IMG_WIDTH, IMG_HEIGHT)))
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    return np.array(resized_digits), labels


def pixels_to_hog_20(img_array):
    hog_featuresData = []
    for img in img_array:
        fd = hog(img,
                 orientations=10,
                 pixels_per_cell=(5,5),
                 cells_per_block=(1,1),
                 visualise=False)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    return np.float32(hog_features)

#define a custom model in a similar class wrapper with train and predict methods
class KNN_MODEL():
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()

class SVM_MODEL():
    def __init__(self, num_feats, C = 1, gamma = 0.1):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF) #SVM_LINEAR, SVM_RBF
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.features = num_feats

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        results = self.model.predict(samples.reshape(-1,self.features))
        return results[1].ravel()


def proc_user_img(img_file, model):
    print('loading "%s for digit recognition" ...' % img_file)
    im = cv2.imread(img_file)
    blank_image = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
    blank_image.fill(255)

    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    plt.imshow(imgray)

    im_digit = (255-imgray)
    im_digit = imresize(im_digit,(IMG_WIDTH ,IMG_HEIGHT))

    hog_img_data = pixels_to_hog_20([im_digit])
    pred = model.predict(hog_img_data)

    return int(pred[0])
    # plt.imshow(im)
    # cv2.imwrite("original_overlay.png",im)
    # cv2.imwrite("final_digits.png",blank_image)
    # cv2.destroyAllWindows()


def get_contour_precedence(contour, cols):
    return contour[1] * cols + contour[0]  #row-wise ordering

def train_models(data_path):
    # TRAIN_DATA_IMG = 'img/digits.png'
    # USER_IMG = 'img/test-4.bmp'

    digits, labels = load_digits(data_path)  # original MNIST data

    print('train data shape', digits.shape)
    print('test data shape', labels.shape)

    digits, labels = shuffle(digits, labels, random_state=256)
    train_digits_data = pixels_to_hog_20(digits)
    X_train, X_test, y_train, y_test = train_test_split(train_digits_data, labels, test_size=0.33, random_state=42)

    model_knn = KNN_MODEL(k=4)
    model_knn.train(train_digits_data, labels)

    model_svm = SVM_MODEL(num_feats=train_digits_data.shape[1])
    model_svm.train(X_train, y_train)

    return model_knn, model_svm


def predict_number(model_knn, model_svm, img_path):
    out_knn = proc_user_img(img_path, model_knn)
    out_svm = proc_user_img(img_path, model_svm)

    return out_knn, out_svm

# #------------------data preparation--------------------------------------------
#
# TRAIN_DATA_IMG = 'img/digits.png'
# USER_IMG = 'img/test-4.bmp'
#
# digits, labels = load_digits(TRAIN_DATA_IMG) #original MNIST data
#
# print('train data shape',digits.shape)
# print('test data shape',labels.shape)
#
# digits, labels = shuffle(digits, labels, random_state=256)
# train_digits_data = pixels_to_hog_20(digits)
# X_train, X_test, y_train, y_test = train_test_split(train_digits_data, labels, test_size=0.33, random_state=42)
#
# #------------------training and testing----------------------------------------
#
# model = KNN_MODEL(k = 3)
# model.train(X_train, y_train)
# preds = model.predict(X_test)
# print('Accuracy: ',accuracy_score(y_test, preds))
#
# model = KNN_MODEL(k = 4)
# model.train(train_digits_data, labels)
# out = proc_user_img(USER_IMG, model)
# print("KNN: ", out)
#
#
#
# model = SVM_MODEL(num_feats = train_digits_data.shape[1])
# model.train(X_train, y_train)
# preds = model.predict(X_test)
# print('Accuracy: ',accuracy_score(y_test, preds))
#
# model = SVM_MODEL(num_feats = train_digits_data.shape[1])
# model.train(train_digits_data, labels)
# out = proc_user_img(USER_IMG, model)
# print("SVM: ", out)

#------------------------------------------------------------------------------