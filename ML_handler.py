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
ML_DIR = "ML"
IMG_PATH = os.path.join(BASE_DIR, IN_DIR, IMG_DIR)
JSON_PATH = os.path.join(BASE_DIR, IN_DIR, JSON_FILE)


TRAINING_PATH = os.path.join(BASE_DIR, ML_DIR, IMG_DIR, "digits.png")

if not os.path.isfile(TRAINING_PATH):
    raise FileNotFoundError(TRAINING_PATH + " doesn't exist")

# load json
try:
    data = json.load(open(JSON_PATH))
except FileNotFoundError("Json File isn't created, Please run setup.py"):
    sys.exit(1)


img_list = data["body"]["images"]

knn, svm = train_models(TRAINING_PATH)
for counter, image in enumerate(img_list):
    pred_svm = 0
    pred_knn = 0
    for file in os.listdir(image["folderPath"]):
        path = os.path.join(image["folderPath"], file)
        if os.path.isdir(path) or "thresh" in file:
            continue
        k, s = predict_number(knn, svm, path)
        pred_knn *= 10
        pred_svm *= 10
        pred_knn += k
        pred_svm += s
    print(pred_knn, pred_svm)
    data["body"]["images"][counter -1]["possibleValues"].append(pred_knn)
    data["body"]["images"][counter -1]["possibleValues"].append(pred_svm)


string = json.dumps(data, indent=4)
out = open(JSON_PATH, "w")
out.write(string)
out.close()

# knn, svm = train_models(TRAINING_PATH)
# out, _ = predict_number(knn, svm)
#
# print(out)