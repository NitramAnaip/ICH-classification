import csv
import cv2
import os
from random import shuffle, randint
import numpy as np
from tensorflow.python.keras import preprocessing 



root="/home/ubuntu/martin/Behold/"
path=root + "data/behold_coding_challenge_train.csv"

def create_label_list():
	label_list=[]
	with open(path) as csvfile:
		data=csv.reader(csvfile, delimiter=',')
		for line in data:
			element=[line[1], [line[i] for i in range (2,5)]]
			label_list.append(element)
	label_list.pop(0) #simply to remove the first line which isn't data
	for elem in label_list:
		for i in range (3):
			elem[1][i]=int(elem[1][i]) 
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

def balancing_data(label_list, train_percent):
	list_types=[[], [], [], [], [], [], [], []]
	for i in range(len(label_list)):
		if label_list[i][1][0]==0 and label_list[i][1][1]==0 and label_list[i][1][2]==0:
			list_types[0].append(label_list[i])
		if label_list[i][1][0]==1 and label_list[i][1][1]==0 and label_list[i][1][2]==0:
			list_types[1].append(label_list[i])
		if label_list[i][1][0]==0 and label_list[i][1][1]==1 and label_list[i][1][2]==0:
			list_types[2].append(label_list[i])
		if label_list[i][1][0]==0 and label_list[i][1][1]==0 and label_list[i][1][2]==1:
			list_types[3].append(label_list[i])
		if label_list[i][1][0]==1 and label_list[i][1][1]==1 and label_list[i][1][2]==0:
			list_types[4].append(label_list[i])
		if label_list[i][1][0]==0 and label_list[i][1][1]==1 and label_list[i][1][2]==1:
			list_types[5].append(label_list[i])
		if label_list[i][1][0]==1 and label_list[i][1][1]==0 and label_list[i][1][2]==1:
			list_types[6].append(label_list[i])
		if label_list[i][1][0]==1 and label_list[i][1][1]==1 and label_list[i][1][2]==1:
			list_types[7].append(label_list[i])
	train_list=[]
	validation_list=[]
	for i in range (8):
		train_index=int(len(list_types[i])*train_percent)
		train_list=train_list+list_types[i][:train_index]
		validation_list=validation_list+list_types[i][train_index:]

	print(len(validation_list), len(train_list), len(label_list))
	return  validation_list, train_list



def output_formater(data):
	for i in range (len(data)):
		output=[]
		for j in range (len(data[i][1])):
			if data[i][1][j]==0:
				output=output+[0,1]
			elif data[i][1][j]==1:
				output=output+[1,0]
		data[i][1]=output
	return 0



class Dataloader():
	def __init__(self, batch_size, train_percent):
		"""
		train_percent is the proportion of our data we use for training (as opposed to validation)
		"""
		self.batch_size=batch_size
		label_list=create_label_list()

		self.validation_list, self.label_list=balancing_data(label_list, train_percent)
		# output_formater(self.validation_list)
		# output_formater(self.label_list)


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
					img_path=root+"data/train_images/train_images/"+self.label_list[indexes[j]][0]+".png"
					
					img=cv2.imread(img_path)
					img=colour_to_grey(img)
					#cv2.imwrite("/home/ubuntu/Desktop/test_img/"+self.label_list[j][0]+"grey.png", img)
					mod_type=randint(0,4)
					dataGenerator(img, mod_type, self.datagen)
					batch.append(img)
					outputs.append(np.array(self.label_list[indexes[j]][1]))
				batch=np.array(batch)
				outputs=np.array(outputs)
				yield(batch, outputs)



			batch=[]
			outputs=[]
			for j in range (nbr_of_batches*self.batch_size, len(indexes)):

				img_path=root+"data/train_images/train_images/"+self.label_list[indexes[j]][0]+".png"
				img=cv2.imread(img_path)
				img=colour_to_grey(img)
				batch.append(img)
				outputs.append(np.array(self.label_list[indexes[j]][1]))
			batch=np.array(batch)
			outputs=np.array(outputs)
			yield(batch, outputs)

	def validation_batch_yielder(self):
		while True:
			indexes=list(range(len(self.validation_list)))
			nbr_of_batches=len(indexes)//self.batch_size
			for i in range (nbr_of_batches):
				start=i*self.batch_size
				end=(i+1)*self.batch_size
				
				batch=[]
				outputs=[]
				for j in range (start, end):
					img_path=root+"data/train_images/train_images/"+self.validation_list[j][0]+".png"
					
					img=cv2.imread(img_path)
					img=colour_to_grey(img)
					batch.append(img)
					outputs.append(np.array(self.validation_list[j][1]))
				batch=np.array(batch)
				outputs=np.array(outputs)
				yield(batch, outputs)
			batch=[]
			outputs=[]
			for j in range (nbr_of_batches*self.batch_size, len(indexes)):

				img_path=root+"data/train_images/train_images/"+self.validation_list[j][0]+".png"
				img=cv2.imread(img_path)
				img=colour_to_grey(img)
				batch.append(img)
				outputs.append(np.array(self.validation_list[j][1]))
			batch=np.array(batch)
			outputs=np.array(outputs)
			yield(batch, outputs)


	def test_batch_yielder(self):
		test_img_paths=os.listdir(root+"data/test_images/test_images")
		for j in range (len(test_img_paths)):

			batch=[]
			img_path=root+"data/test_images/test_images/"+test_img_paths[j]
			img=cv2.imread(img_path)
			img=colour_to_grey(img) #we have to turn it to grey img since tht is what our network was trained on
			batch.append(img)
			batch=np.array(batch)
			yield(batch)
		
		
