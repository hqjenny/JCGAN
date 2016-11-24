
from utils import *
import os
import numpy as np

identity_matrix = np.identity(3)
# apply a random filter that perturbs the color of object images
constant_filter = identity_matrix + np.random.uniform(0, 1, identity_matrix.shape)*.5
#constant_filter = identity_matrix + 0.05


with open("obj.txt") as f:
    real_data = f.read().splitlines()

    obj_batch = [get_image(real_data[0], 256, resize_w=256)] 
data_type= np.float32
batch_images = np.array(obj_batch).astype(data_type)
#print(batch_images)
show_image(batch_images[0])
#show_input_triplet(batch_images, batch_images, batch_images)
shape = batch_images.shape
batch_images = batch_images.reshape( [shape[0], shape[1] * shape[2], shape[3]])

obj_batch_images = np.reshape(np.dot(batch_images, constant_filter), shape)
# (1 + 0.5 * 3)  
max_v = np.max(obj_batch_images) 
min_v = np.min(obj_batch_images) 

scale = max(max_v, abs(min_v)) 
obj_batch_images /= scale
print(obj_batch_images)
show_image(obj_batch_images[0])
