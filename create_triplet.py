from glob import glob
import os
import random
import scipy.misc
import matplotlib.pyplot as plt

dataset = "web_static_outdoor_street_freiberg_germany"

#obj_data = glob(os.path.join("./data/images/", dataset, "*.jpg"))
mask_data = glob(os.path.join("./data/masks/", dataset, "*.jpg"))
bg_data = glob(os.path.join("./data/images/", dataset, "*.jpg"))

target = open("triplet.txt", 'w')
obj = [ x.replace("/masks/","/images/") for x in mask_data]
target = open("obj.txt", 'w')
[target.write("%s\n"%name) for name in obj]
target.close()

target = open("mask.txt", 'w')
[target.write("%s\n"%name) for name in mask_data]
target.close()

target = open("bg.txt", 'w')
random.shuffle(bg_data)
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
