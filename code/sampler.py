import numpy as np
import csv
import cv2
from utils import colour_to_grey, create_label_list
from random import shuffle, randint
from math import ceil
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras import preprocessing



root = "/home/ubuntu/martin/kaggle/"
path = root + "data/kaggle_coding_challenge_train.csv"




def data_visualiser(labels_list):
    #prints the repartition in the different classes
    classes=[[0,0], [0,0], [0,0]]
    for i in range (len(labels_list)):
        for classe in range (0,3):
            if labels_list[i][1][classe]==0:
                classes[classe][0]+=1
            else:
                classes[classe][1]+=1
    return classes







def dataGenerator(img, mod_type, datagen):
    """
    data augmenting function
    """
    if mod_type == 0:
        img = datagen.apply_transform(
            x=img, transform_parameters={"flip_horizontal": True}
        )
    elif mod_type == 1:
        img = datagen.apply_transform(
            x=img, transform_parameters={"flip_vertical": True}
        )
    elif mod_type == 2:
        img = datagen.apply_transform(x=img, transform_parameters={"tx": 5})
    elif mod_type == 3:
        img = datagen.apply_transform(x=img, transform_parameters={"ty": 5})
    elif mod_type == 4:
        img = datagen.apply_transform(x=img, transform_parameters={"ty": -5})
    elif mod_type == 5:
        img = datagen.apply_transform(x=img, transform_parameters={"tx": -5})
    elif mod_type == 5:
        img = datagen.apply_transform(x=img, transform_parameters={"theta": 90})
    elif mod_type == 6:
        img = datagen.apply_transform(x=img, transform_parameters={"theta": 270})
    elif mod_type == 7:
        return img
    return img




class DataLoader():
    def __init__(self, classe, percentage):
        """
        I am going to make binary classification on each of thye three classes one after the other.
        classe is the class I'm looking at in this data
        can take value 0, 1 or 2
        percentage is a list of form [0.8, 0.1, 0.1] for instance. It says: 80% training data, 10% validation, 10% test
        """
        print("start init dataloader")
        self.percentage=percentage
        self.classe=classe
        self.label_list=create_label_list()
        self.train_data=[]
        self.valid_data=[]
        self.test_data=[]
        print("end init dataloader")

    def balance_data(self):
        list_types = [[], []]
        for i in range(len(self.label_list)):
            if (self.label_list[i][1][self.classe] == 1):
                list_types[0].append([self.label_list[i][0], [0,1]])
            else:
                list_types[1].append([self.label_list[i][0], [1,0]])
        for i in range (len(list_types)):
            self.train_data=self.train_data+list_types[i][: int(len(list_types[i])*self.percentage[0])]
            self.valid_data=self.valid_data+list_types[i][int(len(list_types[i])*self.percentage[0]) : int(len(list_types[i])*(self.percentage[0]+self.percentage[1]))]
            self.test_data=self.test_data+list_types[i][int(len(list_types[i])*(self.percentage[0]+self.percentage[1])):]
        print("end of balancing data")



















class Sampler(Sequence):
    def __init__(self, data, batch_size, data_type, network):
        print("start init")
        self.datagen = preprocessing.image.ImageDataGenerator()
        self.batch_size=batch_size
        self.data_type=data_type #can be "train" or "valid"
        if self.data_type=="test":
            self.batch_size=1
        self.network=network
        self.data=data
        self.iterations = ceil(len(self.data) / self.batch_size)
        self.indexes=list(range(len(data)))
        shuffle(self.indexes)
        print("end init")



    def __getitem__(self, idx):
        start=idx*self.batch_size
        end=(idx+1)*self.batch_size
        batch=[]
        outputs=[]
        for index in self.indexes[start:end]:
            img_path = (
                        root
                        + "data/train_images/train_images/"
                        + self.data[index][0]
                        + ".png"
                    )
            img = cv2.imread(img_path)
            if self.network!="mobilenet":
                img = colour_to_grey(img)
            if self.data_type=="train":
                mod_type =randint(0, 7) #randomly choose a transformation to apply to the image at hand
                img=dataGenerator(img, mod_type, self.datagen)

            batch.append(img)
            outputs.append(self.data[index][1])

        batch = np.array(batch)
        outputs = np.array(outputs)
        return (batch, outputs)
        


    def __len__(self):
        return self.iterations

    def __hash__(self):
        return hash(repr(self))

    def on_epoch_end(self):
        if self.data_type=='train':
            shuffle(self.indexes)






def test_generator(test_data, network):
    for i in range (len(test_data)):
        img_path=(
                    root
                    + "data/train_images/train_images/"
                    + test_data[i][0]
                    + ".png"
                )
        img = cv2.imread(img_path)
        if network!="mobilenet":
            img = colour_to_grey(img)
        yield(np.array([img]))






        

