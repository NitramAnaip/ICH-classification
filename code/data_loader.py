import csv
import cv2
import os
from random import shuffle, randint
import numpy as np
from tensorflow.python.keras import preprocessing 

"""
in lack of a better word element here will design an instance of a brain image and/or its output
"""

root="/home/ubuntu/Desktop/Stages/exos_stages/Behold/data/"
path=root + "behold_coding_challenge_train.csv"

def create_label_list():
	label_list=[]
	with open(path) as csvfile:
		data=csv.reader(csvfile, delimiter=',')
		for line in data:
			element=[line[1], [line[i] for i in range (2,5)]]
			label_list.append(element)
	label_list.pop(0) #simply to remove the first line which isn't data
	return label_list




def colour_to_grey(img):
	"""
	This function is to reduce the size of inputs we put in  the CNN. We're not loosing any information
	"""
	w, h=img.shape[:2]
	grey=np.zeros((w, h, 1))
	grey[:,:,0]=img[:,:,0]
	del img
	return grey

def dataGenerator(img, mod_type, datagen):
	if mod_type==0:
		img=datagen.apply_transform(x=img, transform_parameters={'flip_horizontal':True})
	elif mod_type==1:
		img=datagen.apply_transform(x=img, transform_parameters={'flip_vertical':True})
	elif mod_type==2:
		img=datagen.apply_transform(x=img, transform_parameters={'tx':5})
	if mod_type==3:
		img=datagen.apply_transform(x=img, transform_parameters={'ty':5})
	if mod_type==4:
		return 0
	return 0



class Dataloader():
	def __init__(self, batch_size, train, train_percent):
		"""
		if train is true we're creating batches for training. If train is false we're generating validation batches 
		train_percent is the proportion of our data we use for training (as opposed to validation)
		"""
		self.batch_size=batch_size
		self.label_list=create_label_list()
		train_index=int(train_percent*len(self.label_list))
		if train:
			self.label_list=self.label_list[:train_index]

		else:
			self.label_list=self.label_list[train_index:]

		self.datagen = preprocessing.image.ImageDataGenerator()

		


	def batch_yielder(self):
		while True:
			indexes=list(range(len(self.label_list)))
			shuffle(indexes)
			nbr_of_batches=len(indexes)//self.batch_size
			for i in range (nbr_of_batches):
				start=i*self.batch_size
				end=(i+1)*self.batch_size
				
				batch=[]
				outputs=[]
				for j in range (start, end):
					img_path=root+"train_images/train_images/"+self.label_list[j][0]+".png"
					
					img=cv2.imread(img_path)
					img=colour_to_grey(img)
					#cv2.imwrite("/home/ubuntu/Desktop/test_img/"+self.label_list[j][0]+"grey.png", img)
					mod_type=randint(0,4)
					dataGenerator(img, mod_type, self.datagen)
					batch.append(img)
					outputs.append(np.array(self.label_list[j][1]))
				batch=np.array(batch)
				outputs=np.array(outputs)
				yield(batch, outputs)
			batch=[]
			outputs=[]
			for j in range (nbr_of_batches*self.batch_size, len(indexes)):

				img_path=root+"train_images/train_images/"+self.label_list[j][0]+".png"
				img=cv2.imread(img_path)
				img=colour_to_grey(img)
				batch.append(img)
				outputs.append(np.array(self.label_list[j][1]))
			batch=np.array(batch)
			outputs=np.array(outputs)
			yield(batch, outputs)

	def test_batch_yielder(self):
		test_img_paths=os.listdir(root+"test_images/test_images")
		nbr_of_batches=len(test_img_paths)//self.batch_size
		for i in range (nbr_of_batches):
			start=i*self.batch_size
			end=(i+1)*self.batch_size
			
			batch=[]
			for j in range (start, end):
				img_path=root+"test_images/test_images/"+test_img_paths[j]
				img=cv2.imread(img_path)
				img=colour_to_grey(img) #we have to turn it to grey img since tht is what our network was trained on
				batch.append(img)
			batch=np.array(batch)
			yield(batch)
		batch=[]
		for j in range (nbr_of_batches*self.batch_size, len(test_img_paths)):
			img_path=root+"test_images/test_images/"+test_img_paths[j]
			img=cv2.imread(img_path)
			img=colour_to_grey(img) 
			batch.append(img)

		batch=np.array(batch)
		yield(batch)


