#-*- coding:utf-8 –*-
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import mode
import tensorflow as tf

class Dataset(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def tra_te(self,lp):
        #lp labeled proportion 已标记数据的比例
        size = self.x.shape[0]
        #打乱顺序
        perm = np.arange(size)
        np.random.shuffle(perm)

        self._x = self.x[perm]
        self._y = self.y[perm]

        # 1.将数据分为训练集与测试集
        self.x_train = self._x[0:1500]
        self.y_train = self._y[0:1500]

        x_test = self._x[1500:size]
        y_test = self._y[1500:size]

        #2.将训练数据分为已标记数据与未标记数据
        size2 = self.x_train.shape[0]
        lpsize = round(size2*lp)
        x_labeled = self.x_train[0:lpsize]
        y_labeled = self.y_train[0:lpsize]
        x_unlabeled0 = self.x_train[lpsize:size2]
        # y_unlabeled0 = self.y_train[lpsize:size2]

        #3.对训练数据进行GMM聚类
        gmm = GaussianMixture(n_components=10, covariance_type='full', tol=1e-6, max_iter=1000)
        gmm.fit(self.x_train)

        #4.标签传递
        y_unlabeled = gmm.predict(x_unlabeled0)
        for i in range(10):
            y_hat = gmm.predict(x_labeled[y_labeled == i])
            print(y_hat)
            zhongshu = mode(y_hat)     #选出众数
            inx = np.argwhere(y_unlabeled == zhongshu)
            y_unlabeled[inx] = i+10
        y_unlabeled-=10

        #传出数据
        x_train1 = np.concatenate([x_labeled,x_unlabeled0])
        y_train1 = np.concatenate([y_labeled,y_unlabeled])
        with tf.Session() as sess:
            y_train1 = sess.run(tf.one_hot(y_train1,depth=10,on_value=1,axis=1))
            y_labeled = sess.run(tf.one_hot(y_labeled,depth=10,on_value=1,axis=1))
            y_test = sess.run(tf.one_hot(y_test,depth=10,on_value=1,axis=1))
            self.y_train = sess.run(tf.one_hot(self.y_train,depth=10,on_value=1,axis=1))

        self.x_train1 = x_train1          #标签传递之后所有的数据
        self.y_train1 = y_train1
        self.x_labeled = x_labeled        #只有已标记数据
        self.y_labeled = y_labeled
        self.x_test = x_test
        self.y_test = y_test


