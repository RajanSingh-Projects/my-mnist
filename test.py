import numpy as np
import sys
import os
import tensorflow as tf
#import matplotlib.pyplot as plt
from PIL import Image
#from Conv_net import conv_net, n_classes



n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

keep_prob = 1.0

x = tf.placeholder(tf.float32, [None, n_input])

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name="wc1"),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name="wc2"),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024]), name="wd1"),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]),name="out")
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name="bc1"),
    'bc2': tf.Variable(tf.random_normal([64]), name="bc2"),
    'bd1': tf.Variable(tf.random_normal([1024]),name="bd1"),
    'out': tf.Variable(tf.random_normal([n_classes]),name="out1")
}
def main():
  arg = sys.argv[1:]
  img = Image.open(arg[0]).resize((28,28), Image.ANTIALIAS).convert('1')
  #img = Image.open('image').resize((28,28), Image.ANTIALIAS).convert('1')

  #img1 = Image.open('myfile.png')
  myarray = np.array(img.getdata(), np.float32)
  myarr = np.reshape(myarray, (1, np.product(myarray.shape)))

  myarr = np.true_divide(myarr, myarr.max())
  pred = conv_net(x, weights, biases, keep_prob)
  prediction=tf.argmax(pred,1)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    # Restore variables from disk
    saver.restore(sess, "./model.ckpt")
    #print("Model restored.")
    print "predictions", prediction.eval(feed_dict={x: myarr})
#  print result
# https://www.classes.cs.uchicago.edu/archive/2013/spring/12300-1/pa/pa1/digit.png
if __name__=="__main__":
  main()










