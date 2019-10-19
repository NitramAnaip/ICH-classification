# import csv
# import cv2
import numpy as np
import matplotlib.pyplot as plt



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
y1=[0,2,4,6,8,9,5,6,5,6,5,6,5,6,5]
y=range(15)
x=range(15)
plt.plot(x,y, y1)
plt.show()

