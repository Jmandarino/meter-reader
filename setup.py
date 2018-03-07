import os
import sys
from json import dumps
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime


def get_date_from_exif(imagePath):
    """Takes a path to and image and return EXIF date

    :param imagePath: path to image file
    :return: str in format: (%m/%d/%y %H:%M)
    """
    try:
        image = Image.open(imagePath)

    except FileNotFoundError:
        return ""
    info = image._getexif()
    date_string = ""

    if info:
        for tag, value in info.items():
            # from number value tags get a string value
            decoded = TAGS.get(tag, tag)
            if decoded == "DateTime":
                date_string = value
                break

    # '2018:02:18 20:50:24'
    date = datetime.strptime(date_string, "%Y:%m:%d %H:%M:%S")

    return date.strftime("%m/%d/%y %H:%M")


"""
Takes files from in/imgs and converts them to a images.json file to be used further in the program
"""

# date = datetime.strptime(x, "%m/%d/%y %H:%M")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = "in"
IMG_DIR = "imgs"
JSON_FILE = "images.json"
IMG_PATH = os.path.join(BASE_DIR, IN_DIR, IMG_DIR)

try:
    images = []
    for fname in os.listdir(IMG_PATH):
        path = os.path.join(IMG_PATH, fname)
        if not os.path.isdir(path):
            images.append(fname)


except FileNotFoundError:
    print(IMG_PATH, " Does not exist")
    sys.exit(1)

# create JSON base
base = {"body": {"images": []}}
image_objs = []
# create image objects
for image in images:
    d = {"imagePath": os.path.join(IMG_PATH, image), "folderPath": IMG_PATH, "name": image, "date": "", "threshPath": "",
         "KNNValue": -1, "SVMValue": -1, "actualValue": -1}

    date = get_date_from_exif(d["imagePath"])
    d["date"] = date

    image_objs.append(d)


base["body"]["images"] = image_objs

string = dumps(base, indent=4)
path = os.path.join(BASE_DIR, IN_DIR, JSON_FILE)
out = open(path, "w")
out.write(string)
out.close()
print("Setup.py was successful")