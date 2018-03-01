import json
import os
import sys
# control flow:
# import images from in/images.json -> digital ocr -> ML -> graph data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = "/in"
IMG_DIR = "/imgs"
JSON_FILE = "/images.json"
IMG_PATH = BASE_DIR + IN_DIR + IMG_DIR
JSON_PATH = BASE_DIR + IN_DIR + JSON_FILE


# load json
try:
    data = json.load(open(JSON_PATH))
except FileNotFoundError:
    print("Json File isn't created, Please run setup.py")
    sys.exit(1)


img_list = data["body"]["images"]


for x in img_list:
    print(x["imagePath"])