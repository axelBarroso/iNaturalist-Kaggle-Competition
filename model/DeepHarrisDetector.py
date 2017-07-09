import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
from utils.tools import *

def DeepHarrisDetector(input):
    """The model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    with tf.variable_scope('conv_derivativeX', reuse=True) as scope:
        conv_dx = tf.nn.conv2d(input,
                                kernel_filter_dx_5,
                                strides=[1, 1, 1, 1],
                                padding='SAME')
        conv_dx_abs = tf.abs(conv_dx)
        maxValue = tf.reduce_max(conv_dx_abs)
        dx_norm = tf.divide(conv_dx_abs, maxValue)
        dx = tf.multiply(dx_norm, 255)

    with tf.variable_scope('conv_derivativeX', reuse=True) as scope:
        conv_dy = tf.nn.conv2d(input,
                            kernel_filter_dy_5,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        conv_dy_abs = tf.abs(conv_dy)
        maxValue = tf.reduce_max(conv_dy_abs)
        dy_norm = tf.divide(conv_dy_abs, maxValue)
        dy = tf.multiply(dy_norm, 255)

    with tf.variable_scope('Ixx', reuse=True) as scope:
        Ixx = tf.multiply(dx, dx)

    with tf.variable_scope('Iyy', reuse=True) as scope:
        Iyy = tf.multiply(dy, dy)

    with tf.variable_scope('Ixy', reuse=True) as scope:
        Ixy = tf.multiply(dx, dy)

    with tf.variable_scope('Sxx', reuse=True) as scope:
        Sxx = tf.nn.conv2d(Ixx,
                           gaussian_filter_5,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

    with tf.variable_scope('Syy', reuse=True) as scope:
        Syy = tf.nn.conv2d(Iyy,
                            gaussian_filter_5,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

    with tf.variable_scope('Sxy', reuse=True) as scope:
        Sxy = tf.nn.conv2d(Ixy,
                            gaussian_filter_5,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

    with tf.variable_scope('R', reuse=True) as scope:

        with tf.variable_scope('det', reuse=True) as scope:
            det = tf.subtract(tf.multiply(Sxx, Syy), tf.multiply(Sxy, Sxy))

        with tf.variable_scope('tr', reuse=True) as scope:
            tr = tf.add(Sxx, Syy)

        tr_term = tf.multiply(k, tf.multiply(tr, tr))
        R_map = tf.subtract(det, tr_term)

    with tf.variable_scope('max_windows_response', reuse=True) as scope:
        cond = tf.less(0.0, R_map)
        scores = tf.where(cond, R_map, tf.zeros(tf.shape(R_map)))


        response_window_map = tf.nn.conv2d(scores, normfilter, strides=[1, 1, 1, 1], padding='SAME')
        response_window_map_2 = tf.nn.conv2d(response_window_map, normfilter_2, strides=[1, 1, 1, 1], padding='SAME')

    return tf.squeeze(response_window_map_2)