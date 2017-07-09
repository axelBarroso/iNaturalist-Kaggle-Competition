import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from data_providers.utils import *
from model.DeepHarrisDetector import *

shape = (None, cf.image_shape[0], cf.image_shape[1], 1)

class harris_class():
    def __init__(self):
        with tf.name_scope('inputs'):
            self.images = tf.placeholder(tf.float32, shape=shape, name='input_images')

            self.harris_scores_maps = DeepHarrisDetector(self.images)

            # Create a session for running Ops on the Graph.
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            # Create the op for initializing variables.
            init_op = tf.global_variables_initializer()

            # Run the Op to initialize the variables.
            self.sess.run(init_op)

            self.fetches = None
            self.maps = None

    def Harris_score_Map(self, inputs):
        feed_dict = {
            self.images: inputs,
        }

        self.fetches = [self.harris_scores_maps]
        self.maps = self.sess.run(self.fetches, feed_dict=feed_dict)

        return self.maps[0]