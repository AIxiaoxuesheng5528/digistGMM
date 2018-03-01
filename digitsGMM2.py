#-*- coding:utf-8 –*-
from sklearn import datasets
import tensorflow as tf
from Dataset import Dataset
#训练数据与测试数据的加载
digits = datasets.load_digits()
x_train0 = digits.data
y_train0 = digits.target
data = Dataset(x_train0,y_train0)
data.tra_te(0.1)

##入口参数
x = tf.placeholder("float", shape=[None, 64])
y_ = tf.placeholder("float", shape=[None, 10])
keep_prob = tf.placeholder("float")

#定义卷积与池化

#通过函数的形式定义权重变量。变量的初始值，来自于截取正态分布中的数据。
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#通过函数的形式定义偏置量变量，偏置的初始值都是0.1，形状由shape定义。
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#定义卷积函数，其中x是输入，W是权重，也可以理解成卷积核，strides表示步长，或者说是滑动速率，包含长宽方向
#的步长。padding表示补齐数据。 目前有两种补齐方式，一种是SAME，表示补齐操作后（在原始图像周围补充0），实
#际卷积中，参与计算的原始图像数据都会参与。一种是VALID，补齐操作后，进行卷积过程中，原始图片中右边或者底部
#的像素数据可能出现丢弃的情况。
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#这步定义函数进行池化操作，在卷积运算中，是一种数据下采样的操作，降低数据量，聚类数据的有效手段。常见的
#池化操作包含最大值池化和均值池化。这里的2*2池化，就是每4个值中取一个，池化操作的数据区域边缘不重叠。
#函数原型：def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None)。对ksize和strides
#定义的理解要基于data_format进行。默认NHWC，表示4维数据，[batch,height,width,channels]. 下面函数中的ksize，
#strides中，每次处理都是一张图片，对应的处理数据是一个通道（例如，只是黑白图片）。长宽都是2，表明是2*2的
#池化区域，也反应出下采样的速度。
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#网络的搭建

#一层卷积激活池化
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,8,8,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#二层卷积激活池化
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层
W_fc1 = weight_variable([2 * 2 * 64, 512])
b_fc1 = bias_variable([512])
#将第二层池化后的数据进行变形
h_pool2_flat = tf.reshape(h_pool2, [-1, 2*2*64])
#进行矩阵乘，加偏置后进行relu激活
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#对第二层卷积经过relu后的结果，基于tensor值keep_prob进行保留或者丢弃相关维度上的数据。这个是为了防止过拟合，快速收敛。
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([512, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#实际值y_与预测值y_conv的自然对数求乘积，在对应的维度上上求和，该值作为梯度下降法的输入
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

#准确率计算
#首先分别在训练值y_conv以及实际标签值y_的第一个轴向取最大值，比较是否相等
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#对correct_prediction值进行浮点化转换，然后求均值，得到精度。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#下面基于步长1e-4来求梯度，梯度下降方法为AdamOptimizer。
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(200):
    if i % 10 == 0:
      # 计算精度，通过所取的batch中的图像数据以及标签值还有dropout参数，带入到accuracy定义时所涉及到的相关变量中，进行
      # session的运算，得到一个输出，也就是通过已知的训练图片数据和标签值进行似然估计，然后基于梯度下降，进行权值训练。
      train_accuracy = accuracy.eval(feed_dict={x: data.x_labeled, y_: data.y_labeled, keep_prob: 1.0})
      print("step %d, training accuracy %g" % (i, train_accuracy))
    # 此步主要是用来训练W和bias用的。基于似然估计函数进行梯度下降，收敛后，就等于W和bias都训练好了。
    train_step.run(feed_dict={x: data.x_labeled, y_: data.y_labeled, keep_prob: 0.5})
  print("只用已标记数据训练模型test accuracy %g" % accuracy.eval(feed_dict={x: data.x_test, y_: data.y_test, keep_prob: 1.0}))

  sess.run(tf.global_variables_initializer())
  for i in range(200):
    if i % 10 == 0:
      # 计算精度，通过所取的batch中的图像数据以及标签值还有dropout参数，带入到accuracy定义时所涉及到的相关变量中，进行
      # session的运算，得到一个输出，也就是通过已知的训练图片数据和标签值进行似然估计，然后基于梯度下降，进行权值训练。
      train_accuracy = accuracy.eval(feed_dict={x: data.x_train1, y_: data.y_train1, keep_prob: 1.0})
      print("step %d, training accuracy %g" % (i, train_accuracy))
    # 此步主要是用来训练W和bias用的。基于似然估计函数进行梯度下降，收敛后，就等于W和bias都训练好了。
    train_step.run(feed_dict={x: data.x_train1, y_: data.y_train1, keep_prob: 0.5})
  print("结合GMM训练模型test accuracy %g" % accuracy.eval(feed_dict={x: data.x_test, y_: data.y_test, keep_prob: 1.0}))
