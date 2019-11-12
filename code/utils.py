import  numpy as np
import csv

root = "/home/ubuntu/martin/kaggle/"
path = root + "data/kaggle_coding_challenge_train.csv"


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

def colour_to_grey(img):
    """
	This function is to reduce the size of inputs we put in  the CNN by moving from RGB coding to grey. We're not loosing any information
	"""
    w, h = img.shape[:2]
    grey = np.zeros((w, h, 1))
    grey[:, :, 0] = img[:, :, 0]
    del img
    return grey