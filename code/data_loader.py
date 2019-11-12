import csv
import cv2
import os
from utils import colour_to_grey, create_label_list
from random import shuffle, randint
import numpy as np
from tensorflow.python.keras import preprocessing



root = "/home/ubuntu/martin/kaggle/"
path = root + "data/kaggle_coding_challenge_train.csv"







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


def balancing_data(label_list, train_percent):
    """
    This is to ensure we have a balanced percentage of every class in both the training set and the validation set
    """
    list_types = [[], [], [], [], [], [], [], []]
    for i in range(len(label_list)):
        if (
            label_list[i][1][0] == 0
            and label_list[i][1][1] == 0
            and label_list[i][1][2] == 0
        ):
            list_types[0].append(label_list[i])
        if (
            label_list[i][1][0] == 1
            and label_list[i][1][1] == 0
            and label_list[i][1][2] == 0
        ):
            list_types[1].append(label_list[i])
        if (
            label_list[i][1][0] == 0
            and label_list[i][1][1] == 1
            and label_list[i][1][2] == 0
        ):
            list_types[2].append(label_list[i])
        if (
            label_list[i][1][0] == 0
            and label_list[i][1][1] == 0
            and label_list[i][1][2] == 1
        ):
            list_types[3].append(label_list[i])
        if (
            label_list[i][1][0] == 1
            and label_list[i][1][1] == 1
            and label_list[i][1][2] == 0
        ):
            list_types[4].append(label_list[i])
        if (
            label_list[i][1][0] == 0
            and label_list[i][1][1] == 1
            and label_list[i][1][2] == 1
        ):
            list_types[5].append(label_list[i])
        if (
            label_list[i][1][0] == 1
            and label_list[i][1][1] == 0
            and label_list[i][1][2] == 1
        ):
            list_types[6].append(label_list[i])
        if (
            label_list[i][1][0] == 1
            and label_list[i][1][1] == 1
            and label_list[i][1][2] == 1
        ):
            list_types[7].append(label_list[i])
    train_list = []
    validation_list = []
    for i in range(8):
        train_index = int(len(list_types[i]) * train_percent)
        train_list = train_list + list_types[i][:train_index]
        validation_list = validation_list + list_types[i][train_index:]
    return validation_list, train_list





class Dataloader:
    def __init__(self, batch_size, train_percent, mobilenet):
        """
		train_percent is the proportion of our data we use for training (as opposed to validation)
        mobile is a boolean telling us whether we are using mobileNet or not
		"""
        self.batch_size = batch_size
        self.mobile=mobilenet
        label_list = create_label_list()

        shuffle(label_list)
        train_index = int(len(label_list) * train_percent)
        self.validation_list=label_list[train_index:]
        self.label_list=label_list[:train_index]


        self.datagen = preprocessing.image.ImageDataGenerator()

    def batch_yielder(self):
        """
        batch generator for the training data
        """
        while True:
            indexes = list(range(len(self.label_list)))
            shuffle(indexes)  #shuffling the indexes to have a variety of classes in a batch
            nbr_of_batches = len(indexes) // self.batch_size
            for i in range(nbr_of_batches):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size

                batch = []
                outputs = []
                for j in range(start, end):
                    img_path = (
                        root
                        + "data/train_images/train_images/"
                        + self.label_list[indexes[j]][0]
                        + ".png"
                    )

                    img = cv2.imread(img_path)
                    if not self.mobile:
                        img = colour_to_grey(img)
                    mod_type =randint(0, 7) #randomly choose a transformation to apply to the image at hand
                    img=dataGenerator(img, mod_type, self.datagen)

                    batch.append(img)
                    outputs.append(np.array(self.label_list[indexes[j]][1]))
                batch = np.array(batch)
                outputs = np.array(outputs)
                yield (batch, outputs)

            # A last loop to get the last of the data
            batch = []
            outputs = []
            for j in range(nbr_of_batches * self.batch_size, len(indexes)):

                img_path = (
                    root
                    + "data/train_images/train_images/"
                    + self.label_list[indexes[j]][0]
                    + ".png"
                )
                img = cv2.imread(img_path)
                if not self.mobile:
                    img = colour_to_grey(img)
                mod_type =randint(0, 7) #randomly choose a transformation to apply to the image at hand
                img=dataGenerator(img, mod_type, self.datagen)                
                batch.append(img)
                outputs.append(np.array(self.label_list[indexes[j]][1]))
            batch = np.array(batch)
            outputs = np.array(outputs)
            yield (batch, outputs)

    def validation_batch_yielder(self):
        """
        batch generator for validation data
        This function has practically the same structure as the test_generator only we don't shuffle the indexes nor do we augment the data
        """
        while True:
            indexes = list(range(len(self.validation_list)))
            nbr_of_batches = len(indexes) // self.batch_size
            for i in range(nbr_of_batches):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size

                batch = []
                outputs = []
                for j in range(start, end):
                    img_path = (
                        root
                        + "data/train_images/train_images/"
                        + self.validation_list[j][0]
                        + ".png"
                    )

                    img = cv2.imread(img_path)
                    if not self.mobile:
                        img = colour_to_grey(img)
                    batch.append(img)
                    outputs.append(np.array(self.validation_list[j][1]))
                batch = np.array(batch)
                outputs = np.array(outputs)
                yield (batch, outputs)
            batch = []
            outputs = []
            for j in range(nbr_of_batches * self.batch_size, len(indexes)):

                img_path = (
                    root
                    + "data/train_images/train_images/"
                    + self.validation_list[j][0]
                    + ".png"
                )
                img = cv2.imread(img_path)
                if not self.mobile:
                    img = colour_to_grey(img)
                batch.append(img)
                outputs.append(np.array(self.validation_list[j][1]))
            batch = np.array(batch)
            outputs = np.array(outputs)
            yield (batch, outputs)

    def test_batch_yielder(self):
        """
        generator for test data
        """
        test_img_paths = os.listdir(root + "data/test_images/test_images")
        for j in range(len(test_img_paths)):

            batch = []
            img_path = root + "data/test_images/test_images/" + test_img_paths[j]
            img = cv2.imread(img_path)
            if not self.mobile:
                img = colour_to_grey(img)  # we have to turn it to grey img since tht is what our network was trained on
            batch.append(img)
            batch = np.array(batch)
            yield (batch)



