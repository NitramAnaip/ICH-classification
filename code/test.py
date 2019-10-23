import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.python.keras import preprocessing 

root="/home/ubuntu/martin/Behold/data/"
path=root + "behold_coding_challenge_train.csv"


# a=create_label_list()
# for i in range (1,len(a)):
# 	print(a[i][0])
# 	img=cv2.imread(root+"train_images/train_images/"+a[i][0]+".png")
# 	print(img.shape)
# 	import pdb
# 	pdb.set_trace()
# 	del img
# datagen = preprocessing.image.ImageDataGenerator()
# img=cv2.imread('/home/ubuntu/Stages/exos_stages/Behold/data/train_images/train_images/train_3580.png')
# img=datagen.apply_transform(x=img, transform_parameters={'ty':5})
# cv2.imwrite('/home/ubuntu/test_img/test_flip.png', img)
# del img



def create_label_list():
	label_list=[]
	with open(path) as csvfile:
		data=csv.reader(csvfile, delimiter=',')
		for line in data:
			element=[line[1], [line[i] for i in range (2,5)]]
			import pdb
			label_list.append(element)
	label_list.pop(0) #simply to remove the first line which isn't data
	for elem in label_list:
		for i in range (3):
			elem[1][i]=int(elem[1][i]) 
	return label_list

lis=create_label_list()



types=[0]*8
for i in range(len(lis)):
	if lis[i][1][0]==0 and lis[i][1][1]==0 and lis[i][1][2]==0:
		types[0]+=1
	if lis[i][1][0]==1 and lis[i][1][1]==0 and lis[i][1][2]==0:
		types[1]+=1
	if lis[i][1][0]==0 and lis[i][1][1]==1 and lis[i][1][2]==0:
		types[2]+=1
	if lis[i][1][0]==0 and lis[i][1][1]==0 and lis[i][1][2]==1:
		types[3]+=1
	if lis[i][1][0]==1 and lis[i][1][1]==1 and lis[i][1][2]==0:
		types[4]+=1
	if lis[i][1][0]==0 and lis[i][1][1]==1 and lis[i][1][2]==1:
		types[5]+=1
	if lis[i][1][0]==1 and lis[i][1][1]==0 and lis[i][1][2]==1:
		types[6]+=1
	if lis[i][1][0]==1 and lis[i][1][1]==1 and lis[i][1][2]==1:
		types[7]+=1
print(types)


tata=[]
tata=tata+[5,6]
print(tata)
