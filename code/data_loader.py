import csv
import cv2
from random import shuffle
import numpy as np

"""
in lack of a better word element here will design an instance of a brain image and/or its output
"""

root="/home/ubuntu/Stages/exos_stages/Behold/data/"
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




class Dataloader():
	def __init__(self, batch_size, train, train_percent):
		"""
		if train is true we're creating batches for training. If train is false we're generating validation batches 
		"""
		self.batch_size=batch_size
		self.label_list=create_label_list()
		train_index=int(train_percent*len(self.label_list))
		if train:
			self.label_list=self.label_list[:train_index]

		else:
			self.label_list=self.label_list[train_index:]
		


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


# will have to augment data...

