import os
import scipy.misc
import numpy as np

from model import JCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 4, "Epoch to train [25]")
# Use different learning rate of discriminator and generator 
flags.DEFINE_float("learning_rate_d", 0.00002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("learning_rate_g", 0.00002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("reg", 0.1, "Regularization rate of L2 filter [0.01]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("real_size", 16, "The size of real sample images [np.inf]")
flags.DEFINE_integer("fake_real_ratio", 1, "Train one fake images against fake_real_ratio of real sample images [1]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 256, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "web_static_outdoor_street_freiberg_germany", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("device", "/cpu:0", "Device [/cpu:0, /gpu:0, /gpu:1]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    config = tf.ConfigProto(allow_soft_placement = True)

    with tf.Session(config = config) as sess:
        jcgan = JCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, output_size=FLAGS.output_size, c_dim=FLAGS.c_dim,
               dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir, sample_dir=FLAGS.sample_dir, device=FLAGS.device, reg=FLAGS.reg)

        if FLAGS.is_train:
            jcgan.train(FLAGS)
        else:
            jcgan.load(FLAGS.checkpoint_dir)

        if FLAGS.visualize:
            #to_json("./web/js/layers.js", [jcgan.h0_w, jcgan.h0_b, jcgan.g_bn0],
            #                              [jcgan.h1_w, jcgan.h1_b, jcgan.g_bn1],
            #                              [jcgan.h2_w, jcgan.h2_b, jcgan.g_bn2],
            #                              [jcgan.h3_w, jcgan.h3_b, jcgan.g_bn3],
            #                              [jcgan.h4_w, jcgan.h4_b, None])

            # Below is codes for visualization
            OPTION = 5
            visualize(sess, jcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()
