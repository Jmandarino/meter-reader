import os
import sys
from json import dumps

"""
Takes files from in/imgs and converts them to a images.json file to be used further in the program
"""


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = "/in"
IMG_DIR = "/imgs"
JSON_FILE = "/images.json"
IMG_PATH = BASE_DIR + IN_DIR + IMG_DIR

try:
    images = os.listdir(IMG_PATH)
except FileNotFoundError:
    print(IMG_PATH, " Does not exist")
    sys.exit(1)

"""
{
  "body":{
    "images":[
      { "imagePath":"","threshPath": "", "KNNValue": -1, "SVMValue":-1, "actualValue":-1},
      { "imagePath":"","threshPath": "", "KNNValue": -1, "SVMValue":-1, "actualValue":-1}
    ]
  }
}
"""
# create JSON base
base = {"body": {"images": []}}
image_objs = []
# create image objects
for image in images:
    d = {"imagePath": str(IMG_PATH+"/"+image),"threshPath": "", "KNNValue": -1, "SVMValue":-1, "actualValue":-1}
    image_objs.append(d)


base["body"]["images"] = image_objs

string = dumps(base, indent=4)
out = open(BASE_DIR + IN_DIR + JSON_FILE, "w")
out.write(string)
out.close()
print("Setup.py was successful")







