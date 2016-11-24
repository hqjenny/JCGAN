from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from ops import *
from utils import *

identity_matrix = np.identity(3)
# apply a random filter that perturbs the color of object images
constant_filter = identity_matrix + np.random.uniform(0, 1, identity_matrix.shape)*0.5

class JCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=64, output_size=480,
                 y_dim=None, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, reg=0.01, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None, device="/cpu:0"):
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
        self.device = device
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        self.reg = reg
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

        with tf.device(self.device):
            # 1. Specify the image size
            # images is the real image fed in D
            self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                        name='real_images')

            self.obj_images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                        name='obj_images')

            self.bg_images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                        name='bg_images')

            self.mask_images = tf.placeholder(tf.bool, [self.batch_size] + [self.output_size, self.output_size],
                                        name='mask_images')

            # 2. Build generator and discriminator
            # G: generated images
            # D: sigmoid of D_logits from real images
            # D_: sigmoid of D_logits_ from generated images
            self.G, self.color_filter = self.generator(self.obj_images, self.bg_images, self.mask_images)
            self.D, self.D_logits = self.discriminator(self.images)

            self.synthesize = self.synthesize(self.obj_images, self.bg_images, self.mask_images)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

            self.d_sum = tf.histogram_summary("d", self.D)
            self.d__sum = tf.histogram_summary("d_", self.D_)
            self.G_sum = tf.image_summary("G", self.G)

            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
            self.l2_reg = self.reg * tf.nn.l2_loss(self.color_filter)
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

        # Read list of input images, modify read_data_list() function if input path is changed
        real_data, obj_data, mask_data, bg_data = read_data_list()

        d_optim = tf.train.AdamOptimizer(config.learning_rate_d, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate_g, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        #tf.initialize_all_variables().run()
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary([self.d__sum,
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        # Feed real images to discriminator
        #real_files = real_data[0:min(config.real_size, len(real_data))]
        # TODO for experiment
        real_files = real_data
        real_idxs = min(config.real_size, len(real_files))// config.batch_size

        counter = 1
        start_time = time.time()

        # if self.load(self.checkpoint_dir):
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")

        # Train the whole dataset for config.epoch times
        for epoch in range(config.epoch):
            batch_idxs = min(len(obj_data), config.train_size) // config.batch_size

            real_files_rand = np.random.randint(len(real_files), size=config.batch_size)
            # Iterate through different obj and bg pairs
            for idx in range(0, batch_idxs):

                # Get the images files for the obj and background, cropped and normalized
                obj_batch_images, mask_batch_images, bg_batch_images = self.read_triplet(obj_data, mask_data, bg_data, idx, config.batch_size)

                # apply a random filter
                shape = obj_batch_images.shape
                obj_batch_images_rs = np.reshape(obj_batch_images, [shape[0], shape[1] * shape[2], shape[3]])
                obj_batch_images = np.reshape(np.dot(obj_batch_images_rs, constant_filter), shape)
                # For debugging - show the images
                #show_input_triplet(obj_batch_images[0], mask_batch_images[0], bg_batch_images[0])

                # Train the same object and bg for the generator; while input different real objects for the discriminator
                # Iterate through the real sample images for the discriminator
                #TODO TMP CHANGE
                # for real_idx in range(real_idxs):
                for real_idx in range(1):

                    #real_batch_images = self.read_batch_images(real_files[idx * config.batch_size:(idx+1)* config.batch_size], True, np.float32)
                    real_batch_images = self.read_batch_images([real_files[i] for i in real_files_rand.tolist()], True, np.float32)
                    #show_image(real_batch_images[0])
                    #print (len(real_files))
                    #print (real_files_rand)
                    #test_image = real_batch_images[0].dot(np.array([[ 1.00441265, -0.00219392, -0.00315708],[ 0.01812466,  0.99680835,  0.00701926], [ 0.00638699, -0.00649006,  1.03297222]]))
                    #show_image(test_image)

                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.images: real_batch_images, self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    errD_real = self.d_loss_real.eval({self.images: real_batch_images})
                    errG = self.g_loss.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})

                    regG = self.l2_reg.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_fake_loss: %.8f, d_real_loss: %.8f, g_loss: %.8f, l2_reg: %.8f" \
                        % (epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake, errD_real, errG, regG))

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    errD_real = self.d_loss_real.eval({self.images: real_batch_images})
                    errG = self.g_loss.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    regG = self.l2_reg.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})

                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_fake_loss: %.8f, d_real_loss: %.8f, g_loss: %.8f, l2_reg: %.8f" \
                        % (epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake, errD_real, errG, regG))

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    self.writer.add_summary(summary_str, counter)

                    #synth_image = self.obj_color.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    #color_filter_offset = self.color_filter_offset.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    color_filter = self.color_filter.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    #print(color_filter_offset)
                    print(color_filter)
                  
                    #show_image(synth_image[0])

                    # Feed in data that make evalutation of d_loss_fake possible,
                    # Used to input z a vector of noise, now is three input images
                    errD_fake = self.d_loss_fake.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})
                    errD_real = self.d_loss_real.eval({self.images: real_batch_images})
                    errG = self.g_loss.eval({self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images})

                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_fake_loss: %.8f, d_real_loss: %.8f, g_loss: %.8f" \
                        % (epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake, errD_real, errG))
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake+errD_real, errG))

                    #Only save the images to the sample folder when it is looping through half way of real images or the end.   
                    #if np.mod(counter, 4) == 1:
                    #TODO TMP CHANGE
                    #if real_idx == real_idxs/2 or real_idx == real_idxs-1:
                    if idx == 0:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.synthesize, self.d_loss, self.g_loss],
                            feed_dict={self.obj_images: obj_batch_images, self.bg_images: bg_batch_images, self.mask_images: mask_batch_images, self.images: real_batch_images}
                        )
                        #show_image(samples[0])
                        #TODO 64 is the output size can update to larger number
                        save_images(samples, [4, 4],
                                    './{}/train_{:02d}_{:04d}_{:04d}.png'.format(config.sample_dir, epoch, idx, real_idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    # if np.mod(counter, 500) == 2:
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
        color_filter_offset = tf.reshape(filter, [-1, shape[3], shape[3]])
        #self.color_filter_offset = color_filter_offset
        #identity = np.identity(shape[3])
        #identity_matrix = tf.constant( np.tile(identity, (shape[0], 1)) , dtype=np.float32, shape = [shape[0],shape[3],shape[3]])
        identity_matrix = tf.constant(np.identity(shape[3]), dtype=np.float32, shape = [shape[3],shape[3]])
        color_filter = identity_matrix + color_filter_offset 
        #color_filter = identity_matrix
        self.color_filter = color_filter
        #print color_filter.get_shape()
        #print obj_image.get_shape()

        # Reshape input image to N x (H x W) x C 3-D
        obj_image_rs = tf.reshape(obj_image, [shape[0], shape[1] * shape[2], shape[3]])
        obj_color = tf.reshape(tf.batch_matmul(obj_image_rs, color_filter), shape)
        print (obj_color.get_shape())
        # Generate N images? N x H x W x C
        # Use select to combine obj and bg

        # change mask dimension from N x H x W x 1 to N x H x W x 3
        mask_image_pack = tf.concat(3, [tf.expand_dims(mask_image, 3) for i in range(3)])
        #print mask_image_pack.get_shape()
        syn_image = tf.select(mask_image_pack, obj_color, bg_image)
        self.obj_color = syn_image

        return syn_image, color_filter

    def siamese(self, image, num_class, train=False):
        #filter_dim = 6 in siamese net

        # if shared weights
        #print image.get_shape()
        conv1 = maxpool2d(lrelu(self.g_bn0(conv2d(image, 16, d_h = 1, d_w = 1, name='g_conv0'), train=train)))
        #print conv1.get_shape()
        conv2 = maxpool2d(lrelu(self.g_bn1(conv2d(conv1, 32, d_h = 1, d_w = 1, name='g_conv1'), train=train)))
        #print conv2.get_shape()
        conv3 = maxpool2d(lrelu(self.g_bn2(conv2d(conv2, 64, d_h = 1, d_w = 1, name='g_conv2'), train=train)))

        # Using leaky relu
        fc4 = lrelu(fc(conv3, 64, "g_fc3"))
        fc5 = lrelu(fc(fc4, 32, "g_fc4"))
        # no ReLu
        fc6 = fc(fc5, num_class, "g_fc5")

        return fc6

    def synthesize(self, obj_image, bg_image, mask_image):

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
        color_filter_offset = tf.reshape(filter, [-1, shape[3], shape[3]])
        identity_matrix = tf.constant(np.identity(shape[3]), dtype=np.float32, shape = [shape[3],shape[3]])
        color_filter = identity_matrix + color_filter_offset        #print color_filter.get_shape()
        #print obj_image.get_shape()

        # Reshape input image to N x (H x W) x C 3-D
        obj_image_rs = tf.reshape(obj_image, [shape[0], shape[1] * shape[2], shape[3]])
        obj_color = tf.reshape(tf.batch_matmul(obj_image_rs, color_filter), shape)
        print(obj_color.get_shape())

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

    # Read batch of obj, mask and bg images
    def read_triplet(self, obj_data, mask_data, bg_data, idx, batch_size):
        #print "obj"
        obj_batch_files = obj_data[idx*batch_size:(idx+1)* batch_size]
        obj_batch_images = self.read_batch_images(obj_batch_files, True, np.float32)

        #print "mask"
        mask_batch_files = mask_data[idx*batch_size:(idx+1)*batch_size]
        mask_batch_images = self.read_batch_images(mask_batch_files, False, np.bool)
        #print "bg"
        bg_batch_files = bg_data[idx*batch_size:(idx+1)*batch_size]
        bg_batch_images = self.read_batch_images(bg_batch_files, True, np.float32)

        return (obj_batch_images, mask_batch_images, bg_batch_images)

    # Read batch of data
    def read_batch_images(self, batch_data, is_norm=True, data_type=np.float32):
        print (batch_data)
        obj_batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale, is_norm = is_norm) for batch_file in batch_data]

        if (self.is_grayscale):
            batch_images = np.array(obj_batch).astype(data_type)[:, :, :, None]
        else:
            batch_images = np.array(obj_batch).astype(data_type)
        return batch_images
