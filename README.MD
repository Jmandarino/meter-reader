# Seven segment display reader (Meter Reader)

## Purpose:

To read the data from Coned Electric meters (7 segment displays) and graph the information

### Workflow:

1. place images inside `in/imgs`
2. run setup.py `python setup.py` this creates `images.json`
3. process image data from `images.json` by running `python digit_ocr.py`
3. run data through MNIST and SVM/KNN
4. Graph data and find trends


### TOOLS:
1. opencv
2. sklearn
3. MNIST Dataset
4. Matplotlib 

### MAJOR TODO's :

1. Connect user flows (processing img -> figure out image -> graph)
2. Error Processing for (OCR)
2. Grab EXIF data from picture
3. Error Processing for data (if ocr failed to read data correctly what should we do


### Requirements and how to install

TODO: later

### HOW TO USE:

1. place images into in/imgs
2. run setup.py



##Links of Interest:
https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/

https://github.com/pavitrakumar78/Python-Custom-Digit-Recognition