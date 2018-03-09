import json
import os
import sys
from ML import train_models, predict_number
# control flow:
# import images from in/images.json -> digital ocr -> ML -> graph data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = "in"
IMG_DIR = "imgs"
JSON_FILE = "images.json"
IMG_PATH = os.path.join(BASE_DIR, IN_DIR, IMG_DIR)
JSON_PATH = os.path.join(BASE_DIR, IN_DIR, JSON_FILE)


# load json
try:
    data = json.load(open(JSON_PATH))
except FileNotFoundError("Json File isn't created, Please run setup.py"):
    sys.exit(1)


img_list = data["body"]["images"]


for x in img_list:
    print(x["imagePath"])

knn, svm = train_models("C:\\Users\\joeym\\PycharmProjects\\meter-reader\\ML\\img")