from glob import glob
import os
import random
import scipy.misc
import matplotlib.pyplot as plt

#parent_folder="./data/images/"
#dataset = "web_static_outdoor_street_freiberg_germany"

# Specify the valid dataset in the path 
parent_folder="/media/cecilia/DATA/cecilia/labelme_car_all/"
real_image_parent_folder="/media/cecilia/DATA/cecilia/labelme_png/"

with open("car_list.txt") as f:
    datasets = f.read().splitlines()
datasets = datasets[0:-1]
obj_data = []
mask_data = []
bg_data = []
real_data = []

for dataset in datasets:
    obj_data.extend(glob(os.path.join(parent_folder + "images_filter", dataset, "*")))
    mask_data.extend(glob(os.path.join(parent_folder + "masks_filter", dataset, "*")))
    bg_data.extend(glob(os.path.join(parent_folder + "images_filter", dataset, "*")))


# with open("real_png.txt") as f:
#     real_image_datasets = f.read().splitlines()
# real_data.extend(glob(os.path.join(parent_folder + "images", dataset, "*")))

# target = open("real_png.txt", 'w')
# [target.write("%s\n"%name) for name in obj_data]
# target.close()

obj = [ x.replace("/masks_filter/","/images_filter/") for x in mask_data]
target = open("obj.txt", 'w')
[target.write("%s\n"%name) for name in obj]
target.close()

target = open("mask.txt", 'w')
[target.write("%s\n"%name) for name in mask_data]
target.close()

target = open("bg.txt", 'w')
#random.shuffle(bg_data)
#print bg_data
[target.write("%s\n"%name) for name in bg_data]
target.close()


# for bg in bg_data:
#
#     for name in mask_data:
#         line = "%s %s %s\n"%(bg, name, name)
#         target.write(line )
#
# target.close()
#
#
# target = open("obj.txt", 'w')
# target2 = open("mask.txt", 'w')
# for name in mask_data:
#     line = "%s\n"%(name)
#     target.write(line)
#     target2.write(line)
#
# target.close()
# target2.close()
