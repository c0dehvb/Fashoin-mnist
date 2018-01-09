import tensorflow as tf
import numpy as np


class CNN:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def build_model(self):
        x = tf.placeholder('float', [None, 28 * 28])
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        y = tf.placeholder('float', [None, 10])

        # 卷积层 1
        weight_1 = tf.Variable(tf.random_normal((2, 2, 1, 32), stddev=0.1))
        biase_1 = tf.Variable(tf.random_normal([32], stddev=0.1))
        conv_1 = tf.nn.relu(tf.nn.conv2d(x_image, weight_1, strides=[1, 1, 1, 1], padding='SAME') + biase_1)
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 卷积层 2
        weight_2 = tf.Variable(tf.random_normal((2, 2, 32, 64), stddev=0.1))
        biase_2 = tf.Variable(tf.random_normal([64], stddev=0.1))
        conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, weight_2, strides=[1, 1, 1, 1], padding='SAME') + biase_2)
        pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 卷积层 3
        weight_3 = tf.Variable(tf.random_normal((2, 2, 64, 16), stddev=0.1))
        biase_3 = tf.Variable(tf.random_normal([16], stddev=0.1))
        conv_3 = tf.nn.relu(tf.nn.conv2d(pool_2, weight_3, strides=[1, 1, 1, 1], padding='SAME') + biase_3)
        pool_3 = tf.nn.max_pool(conv_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 卷积层 4
        weight_4 = tf.Variable(tf.random_normal((2, 2, 16, 4), stddev=0.1))
        biase_4 = tf.Variable(tf.random_normal([4], stddev=0.1))
        conv_4 = tf.nn.relu(tf.nn.conv2d(pool_3, weight_4, strides=[1, 1, 1, 1], padding='SAME') + biase_4)
        pool_4 = tf.nn.max_pool(conv_4, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        # FC
        fc_input = tf.reshape(pool_4, [-1, 64])

        # hid 1
        weight_hid_1 = tf.Variable(tf.random_normal((64, 64 * 2), stddev=0.1))
        biase_hid_1 = tf.Variable(tf.random_normal((64 * 2,), stddev=0.1))
        hid_1 = tf.nn.leaky_relu(tf.matmul(fc_input, weight_hid_1) + biase_hid_1)

        # hid 2
        weight_hid_2 = tf.Variable(tf.random_normal((128, 30), stddev=0.1))
        biase_hid_2 = tf.Variable(tf.random_normal((30,), stddev=0.1))
        hid_2 = tf.nn.leaky_relu(tf.matmul(hid_1, weight_hid_2) + biase_hid_2)

        # output
        weight_hid_3 = tf.Variable(tf.random_normal((30, 10), stddev=0.1))
        biase_hid_3 = tf.Variable(tf.random_normal((10,), stddev=0.1))
        out = tf.matmul(hid_2, weight_hid_3) + biase_hid_3
        softmax = tf.nn.softmax(out)
        predict = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y)
        # 训练模型
        optm = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(predict)
        # 评估模型
        correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return x, y, predict, optm, accuracy


cnn = CNN(1e-7 * 2)
x, y, predict, optm, accuracy = cnn.build_model()

import pandas as pd

df = pd.read_csv("data/fashion-mnist_test.csv")
images = df.values[:, 1:]
labels = df.values[:, 0]

from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()
lb.fit(labels)
labels = lb.transform(labels)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ckpt_dir = "checkpoint"
    saver.restore(sess, "checkpoint/model-360")
    feeds = {x: images.reshape((-1, 784)), y: labels.reshape(-1, 10)}
    err = 100000000000
    iter_num = 360
    while err > 0.01:
        iter_num += 1
        sess.run(optm, feed_dict=feeds)
        # print("predict=", sess.run(predict, feed_dict=feeds))
        err = 1 - sess.run(accuracy, feed_dict=feeds)
        print("第 %d 次训练，错误率为：%f" % (iter_num, err))
        if iter_num % 10 == 0:
            saver.save(sess, ckpt_dir + '/model', global_step=iter_num)
