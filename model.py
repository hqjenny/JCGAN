from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
#from six.moves import xrange
import matplotlib.pyplot as plt

from ops import *
from utils import *

class JCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=64, sample_size=64, output_size=480,
                 y_dim=None, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None] y is one in JCGAN
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn0 = batch_norm(name='d_bn0')
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    # Build the TF data-flow graph
    def build_model(self):
        # 1. Specify the image size
        # TODO output size is different for different input images
        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                    name='real_images')

        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_size, self.output_size, self.c_dim],
                                        name='sample_images')

        self.obj_images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                    name='obj_images')

        self.bg_images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                    name='bg_images')

        self.mask_images = tf.placeholder(tf.bool , [self.batch_size] + [self.output_size, self.output_size],
                                    name='mask_images')

        # 2. Build generator and discriminator
        # G: generated images
        # D: sigmoid of D_logits from real images
        # D_: sigmoid of D_logits_ from generated images
        self.G = self.generator(self.obj_images, self.bg_images, self.mask_images)
        self.D, self.D_logits = self.discriminator(self.images)

        self.sampler = self.sampler(self.obj_images, self.bg_images, self.mask_images)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.G_sum = tf.image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
        # load data from file
        data = glob(os.path.join("./data/images/", config.dataset, "*.jpg"))
        #np.random.shuffle(data)

        #obj_data = glob(os.path.join("./data/images/", config.dataset, "*.jpg"))
        with open("obj.txt") as f:
            obj_data = f.read().splitlines()
        #mask_data = glob(os.path.join("./data/masks/", config.dataset, "*.jpg"))
        with open("mask.txt") as f:
            mask_data = f.read().splitlines()
        #bg_data = glob(os.path.join("./data/images/", config.dataset, "*.jpg"))
        with open("bg.txt") as f:
            bg_data = f.read().splitlines()

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary([self.d__sum,
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        # Feed sample images to discriminator as real image

        sample_files = data[0:min(self.sample_size, len(data))]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale) for sample_file in sample_files]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            # Set crop to True as the images are of different size
            sample_images = np.array(sample).astype(np.float32)
            print sample_images.shape

        counter = 1
        start_time = time.time()

        # if self.load(self.checkpoint_dir):
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")

        # Train the whole dataset for config.epoch times
        for epoch in xrange(config.epoch):


            batch_idxs = min(len(obj_data), config.train_size) // config.batch_size
            print(batch_idxs)

            # iterate through different obj and bg pairs
            for idx in xrange(0, batch_idxs):
                # get the images files for the obj and background
                obj_batch_images, mask_batch_images, bg_batch_images = self.read_triplet(obj_data, mask_data, bg_data, idx, config.batch_size)

                print obj_batch_images.shape
                plt.figure()
                plt.subplot(221)
                plt.imshow(obj_batch_images[0])
                plt.subplot(222)
                plt.imshow(mask_batch_images[0])
                plt.subplot(223)
                plt.imshow(bg_batch_images[0])
                plt.show()

                # Train the same object and bg for the generator; while input different real objects for the discriminator
                sample_idx = min(len(sample), config.sample_size) // config.batch_size
                print "sample idx"
                print(sample_idx)

                for i in range(sample_idx):
                    #inner loop
                    sample_batch_images = sample_images[i * config.batch_size:(i+1)* config.batch_size]
                    # Update D network
                    # TODO use the object image as the real images
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.images: sample_batch_images, self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    self.writer.add_summary(summary_str, counter)

                    # Feed in data that make evalutation of d_loss_fake possible,
                    # used to input z a vector of noise, now is three input images
                    synth_image = self.obj_color.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    print(np.array(synth_image).shape)
                    show_image(synth_image[0])

                    errD_fake = self.d_loss_fake.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    errD_real = self.d_loss_real.eval({self.images: obj_batch_images})
                    errG = self.g_loss.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})

                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake+errD_real, errG))

                    if np.mod(counter, 2) == 1:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            #TODO changed the sampler function
                            feed_dict={self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images, self.images: sample_batch_images}
                        )
                        save_images(samples, [8, 8],
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                #if np.mod(counter, 500) == 2:
                #    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("d_iscriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn0(conv2d(h0, self.df_dim*1, d_h=1, d_w=1, name='d_h1_conv')))
            h2 = lrelu(self.d_bn1(conv2d(h1, self.df_dim*1, d_h=1, d_w=1, name='d_h2_conv')))
            h3 = lrelu(self.d_bn2(conv2d(h2, self.df_dim*1, d_h=1, d_w=1, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, obj_image, bg_image, mask_image):

        with tf.variable_scope("g_enerator") as scope:
            # Get the obj_image shape Batch x H x W x C
            shape = obj_image.get_shape().as_list()

            # Generate vector of (C x C)
            obj = self.siamese(obj_image, shape[3] * shape[3], train=True)
            tf.get_variable_scope().reuse_variables()
            bg = self.siamese(bg_image, shape[3] * shape[3], train=True)
            # Generates batch_size x (C x C) 2-D filter? so using add instead of concat?
            # TODO Concat 18 x 1 -> ReLU -> FC -> 1 x 9
            #concat_filter = tf.add(obj, bg)

        concat_filter = tf.concat(1, [obj, bg])
        #print "Concat shape"
        #print concat_filter.get_shape()
        filter = lrelu(fc(concat_filter, shape[3] * shape[3], "g_fc_concat"))

        # Affine Layer
        # Reshape output to Batch x C x C
        color_filter = tf.reshape(filter, [-1, shape[3], shape[3]])
        #print color_filter.get_shape()
        #print obj_image.get_shape()

        # Reshape input image to N x (H x W) x C 3-D
        obj_image_rs = tf.reshape(obj_image, [shape[0], shape[1] * shape[2], shape[3]])
        obj_color = tf.reshape(tf.batch_matmul(obj_image_rs, color_filter), shape)
        print obj_color.get_shape()
        # Generate N images? N x H x W x C
        # Use select to combine obj and bg

        # change mask dimension from N x H x W x 1 to N x H x W x 3
        mask_image_pack = tf.concat(3, [tf.expand_dims(mask_image, 3) for i in range(3)])
        #print mask_image_pack.get_shape()
        self.obj_color = obj_color
        syn_image = tf.select(mask_image_pack, obj_color, bg_image)
        return syn_image

    def sampler(self, obj_image, bg_image, mask_image):

        with tf.variable_scope("g_enerator") as scope:
            # Get the obj_image shape Batch x H x W x C
            shape = obj_image.get_shape().as_list()
            tf.get_variable_scope().reuse_variables()

            obj = self.siamese(obj_image, shape[3] * shape[3], train=False)
            bg = self.siamese(bg_image, shape[3] * shape[3], train=False)
        tf.get_variable_scope().reuse_variables()
        # Generates batch_size x (C x C) 2-D filter? so using add instead of concat?
        #concat_filter = tf.add(obj, bg)
        concat_filter = tf.concat(1, [obj, bg])
        filter = lrelu(fc(concat_filter, shape[3] * shape[3], "g_fc_concat"))

        # Affine Layer
        # Reshape output to Batch x C x C
        color_filter = tf.reshape(filter, [-1, shape[3], shape[3]])
        print color_filter.get_shape()
        print obj_image.get_shape()

        # Reshape input image to N x (H x W) x C 3-D
        obj_image_rs = tf.reshape(obj_image, [shape[0], shape[1] * shape[2], shape[3]])
        obj_color = tf.reshape(tf.batch_matmul(obj_image_rs, color_filter), shape)
        print obj_color.get_shape()

        # change mask dimension from N x H x W x 1 to N x H x W x 3
        mask_image_pack = tf.concat(3, [tf.expand_dims(mask_image, 3) for i in range(3)])
        #print mask_image_pack.get_shape()
        # Generate N images? N x H x W x C
        # Use select to combine obj and bg
        syn_image = tf.select(mask_image_pack, obj_color, bg_image)
        return syn_image

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def siamese(self, image, num_class, train=False):
        #filter_dim = 6 in siamese net

        # if shared weights
        #print image.get_shape()
        conv1 = maxpool2d(lrelu(self.g_bn0(conv2d(image, 16, d_h = 1, d_w = 1, name='g_conv0'), train=train)))
        #print conv1.get_shape()
        conv2 = maxpool2d(lrelu(self.g_bn1(conv2d(conv1, 32, d_h = 1, d_w = 1, name='g_conv1'), train=train)))
        #print conv2.get_shape()
        conv3 = maxpool2d(lrelu(self.g_bn2(conv2d(conv2, 64, d_h = 1, d_w = 1, name='g_conv2'), train=train)))

        # Using lrelu might need actual ReLu ??
        fc4 = lrelu(fc(conv3, 64, "g_fc3"))
        fc5 = lrelu(fc(fc4, 32, "g_fc4"))
        # no ReLu
        fc6 = fc(fc5, num_class, "g_fc5")

        return fc6


    def read_triplet(self, obj_data, mask_data, bg_data, idx, batch_size):
        print "obj"
        obj_batch_files = obj_data[idx*batch_size:(idx+1)* batch_size]
        obj_batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale) for batch_file in obj_batch_files]
        print "mask"
        mask_batch_files = mask_data[idx*batch_size:(idx+1)*batch_size]
        mask_batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale, is_norm = False) for batch_file in mask_batch_files]
        print "bg"
        bg_batch_files = bg_data[idx*batch_size:(idx+1)*batch_size]
        bg_batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale) for batch_file in bg_batch_files]

        if (self.is_grayscale):
            obj_batch_images = np.array(obj_batch).astype(np.float32)[:, :, :, None]
            mask_batch_images = np.array(mask_batch).astype(np.bool)[:, :, :, None]
            bg_batch_images = np.array(bg_batch).astype(np.float32)[:, :, :, None]
        else:
            obj_batch_images = np.array(obj_batch).astype(np.float32)
            mask_batch_images = np.array(mask_batch).astype(np.bool)
            bg_batch_images = np.array(bg_batch).astype(np.float32)

        return (obj_batch_images, mask_batch_images, bg_batch_images)
