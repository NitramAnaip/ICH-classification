import numpy as np
import csv





root = "/home/ubuntu/martin/Behold/"
path = root + "data/behold_coding_challenge_train.csv"


def create_label_list():
    """
    Creates a list with all training data. Has the following form: [[train_0001, [0,0,1]], [train_0002, [1,0,0]], ....]
    """
    label_list = []
    with open(path) as csvfile:
        data = csv.reader(csvfile, delimiter=",")
        for line in data:
            element = [line[1], [line[i] for i in range(2, 5)]]
            label_list.append(element)
    label_list.pop(0)  # simply to remove the first line which isn't data
    for elem in label_list:
        for i in range(3):
            elem[1][i] = int(elem[1][i])
    return label_list






class DataLoader():
    def __init__(self, classe, percentage):
        """
        I am going to make binary classification on each of thye three classes one after the other.
        classe is the class I'm looking at in this data
        can take value 0, 1 or 2
        percentage is a list of form [0.8, 0.1, 0.1] for instance. It says: 80% training data, 10% validation, 10% test
        """
        self.percentage=percentage
        self.classe=classe
        self.label_list=create_label_list()
        self.train_data=[]
        self.valid_data=[]
        self.test_data=[]

    def balance_data(self):
        list_types = [[], []]
        for i in range(len(self.label_list)):
            if (self.label_list[i][1][0] == 0):
                list_types[0].append(self.label_list[i])
            else:
                list_types[1].append(self.label_list[i])
        for i in range (len(list_types)):
            self.train_data=self.train_data+list_types[i][: int(len(list_types[i])*percentage[0])]
            self.valid_data=self.valid_data+list_types[i][int(len(list_types[i])*percentage[0]) : int(len(list_types[i])*(percentage[0]+percentage[1]))]
            self.test_data=self.test_data+list_types[i][int(len(list_types[i])*(percentage[0]+percentage[1])):]
















class Sampler():
    def __init__(self, batch_size, data_type):
        self.batch_size=batch_size
        self.data_type=data_type #can be "train", "valid", or "test"


    def __getitem__(self, idx):
        a=4

