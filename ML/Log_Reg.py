from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import cv2



mnist = fetch_mldata('MNIST original', data_home="./ML/data")


from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=0)


plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
    pass
    # plt.subplot(1, 5, index + 1)
    # plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    # plt.title('Training: %i\n' % label, fontsize = 20)

logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_img, train_lbl)
logisticRegr.predict(test_img[0].reshape(1,-1))
predictions = logisticRegr.predict(test_img)

score = logisticRegr.score(test_img, test_lbl)
print(score)


image = cv2.imread("test.bmp")
image2 = cv2.imread("test.bmp")

l = []
l.append(image)
out = logisticRegr.predict(image, (41,22,3))

print(out)

