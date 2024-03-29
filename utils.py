"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from glob import glob
import os
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False, is_norm = True):
    #print (image_path)
    #image = imread(image_path, is_grayscale)
    #image = 255 - image
    #print image.shape
    #plt.imshow(image[...,::-1])
    #plt.imshow(image)
    #plt.show()
    image = imread(image_path, is_grayscale)
    image = transform(image, image_size, is_crop, resize_w, is_norm)
    #scipy.misc.imshow(image)
    #print (image.shape)
    return image

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False, type = np.float32):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(type)
    else:
        return scipy.misc.imread(path).astype(type)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64, is_norm = True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image

    if is_norm:
        return np.array(cropped_image)/127.5 - 1.
    else:
        return cropped_image

def inverse_transform(images):
    
    #return (images+1.)/2.
    return ((images+1.) * 127.5).astype(np.uint8)


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc", 
                        "sy": 1, "sx": 1, 
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv", 
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, jcgan, config, option):

  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, jcgan.z_dim))
    samples = sess.run(jcgan.sampler, feed_dict={jcgan.z: z_sample})
    save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, jcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(jcgan.sampler, feed_dict={jcgan.z: z_sample})
      save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(jcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, jcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(jcgan.sampler, feed_dict={jcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, jcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(jcgan.sampler, feed_dict={jcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, jcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(jcgan.sampler, feed_dict={jcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
  elif option == 5:
    obj_data = glob(os.path.join("./data/images/", config.dataset, "*.jpg"))
    mask_data = glob(os.path.join("./data/masks/", config.dataset, "*.jpg"))

    # TODO Assume bg data is the same as the obj for now
    bg_data = glob(os.path.join("./data/images/", config.dataset, "*.jpg"))
    batch_idxs = min(len(obj_data), config.train_size)
    for idx in batch_idxs:
        obj_batch_files = obj_data[idx]
        obj_batch = [get_image(batch_file, jcgan.image_size, is_crop=jcgan.is_crop, resize_w=jcgan.output_size, is_grayscale = jcgan.is_grayscale) for batch_file in obj_batch_files]

        mask_batch_files = mask_data[idx*config.batch_size:(idx+1)*config.batch_size]
        mask_batch = [get_image(batch_file, jcgan.image_size, is_crop=jcgan.is_crop, resize_w=jcgan.output_size, is_grayscale = jcgan.is_grayscale) for batch_file in mask_batch_files]

        bg_batch_files = bg_data[idx*config.batch_size:(idx+1)*config.batch_size]
        bg_batch = [get_image(batch_file, jcgan.image_size, is_crop=jcgan.is_crop, resize_w=jcgan.output_size, is_grayscale = jcgan.is_grayscale) for batch_file in bg_batch_files]

        if (jcgan.is_grayscale):
            obj_batch_images = np.array(obj_batch).astype(np.float32)[:, :, :, None]
            mask_batch_images = np.array(mask_batch).astype(np.float32)[:, :, :, None]
            bg_batch_images = np.array(bg_batch).astype(np.float32)[:, :, :, None]
        else:
            obj_batch_images = np.array(obj_batch).astype(np.float32)
            mask_batch_images = np.array(mask_batch).astype(np.float32)
            bg_batch_images = np.array(bg_batch).astype(np.float32)
    samples = sess.run(jcgan.sampler, feed_dict={jcgan.z: z_sample})

def show_image(image, is_norm = True, is_show=True):
    if is_norm:
        image = (image + 1.) * 127.5
    image = np.array(image).astype(np.uint8)
    plt.imshow(image)
    if is_show:
        plt.show()

def show_input_triplet(obj_batch_images, mask_batch_images, bg_batch_images):
    for i in range(obj_batch_images.shape[0]):
        #plt.figure()
        plt.subplot(221)
        show_image(obj_batch_images[i],is_show=False)
        plt.subplot(222)
        show_image(mask_batch_images[i],is_show=False)
        plt.subplot(223)
        show_image(bg_batch_images[i],is_show=False)
        plt.show()

# Use create_triplet.py to generate the input list
def read_data_list(real_path="real.txt", obj_path="obj.txt", mask_path="mask.txt", bg_path="bg.txt"):
    # Load data from file
    #real_data = glob(os.path.join("./data/images/", config.dataset, "*.jpg"))
    with open(real_path) as f:
        real_data = f.read().splitlines()
    #np.random.shuffle(data)

    #obj_data = glob(os.path.join("./data/images/", config.dataset, "*.jpg"))
    with open(obj_path) as f:
        obj_data = f.read().splitlines()

    #mask_data = glob(os.path.join("./data/masks/", config.dataset, "*.jpg"))
    with open(mask_path) as f:
        mask_data = f.read().splitlines()
    #bg_data = glob(os.path.join("./data/images/", config.dataset, "*.jpg"))
    with open(bg_path) as f:
        bg_data = f.read().splitlines()

    return real_data, obj_data, mask_data, bg_data

# def read_data_list():
#     # Load data from file
#     #real_data = glob(os.path.join("./data/images/", config.dataset, "*.jpg"))
#     with open("real.txt") as f:
#         real_data = f.read().splitlines()
#     #np.random.shuffle(data)

#     #obj_data = glob(os.path.join("./data/images/", config.dataset, "*.jpg"))
#     with open("obj.txt") as f:
#         obj_data = f.read().splitlines()
#     #mask_data = glob(os.path.join("./data/masks/", config.dataset, "*.jpg"))
#     with open("mask.txt") as f:
#         mask_data = f.read().splitlines()
#     #bg_data = glob(os.path.join("./data/images/", config.dataset, "*.jpg"))
#     with open("bg.txt") as f:
#         bg_data = f.read().splitlines()

#     return (real_data, obj_data, mask_data, bg_data)
