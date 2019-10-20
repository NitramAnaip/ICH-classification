import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import preprocessing 

root="/home/ubuntu/Desktop/Stages/exos_stages/Behold/data/"
path=root + "behold_coding_challenge_train.csv"


# a=create_label_list()
# for i in range (1,len(a)):
# 	print(a[i][0])
# 	img=cv2.imread(root+"train_images/train_images/"+a[i][0]+".png")
# 	print(img.shape)
# 	import pdb
# 	pdb.set_trace()
# 	del img
datagen = preprocessing.image.ImageDataGenerator()
img=cv2.imread('/home/ubuntu/Stages/exos_stages/Behold/data/train_images/train_images/train_3580.png')
img=datagen.apply_transform(x=img, transform_parameters={'ty':5})
cv2.imwrite('/home/ubuntu/test_img/test_flip.png', img)
del img

