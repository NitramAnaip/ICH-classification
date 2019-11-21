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



def from_proba_to_output(probabilities, threshold):
    outputs = np.copy(probabilities)
    for i in range(len(outputs)):

        if (float(outputs[i])) > threshold:
            outputs[i] = int(1)
        else:
            outputs[i] = int(0)
    return np.array(outputs)

def tranf_for_conf_matrix(probabilities, threshold):
    outputs=[]
    for i in range (len(probabilities)):
        if probabilities[i]>threshold:
            outputs.append(1)
        else:
            outputs.append(0)
    return outputs


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
