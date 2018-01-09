import pandas as pd
import cv2

df = pd.read_csv("data/fashion-mnist_test.csv")
images = df.values[:, 1:]

for i in range(images.shape[0]):
    cv2.imwrite("data/imgs/test-%d.jpg" % i, images[i].reshape((28,28)))