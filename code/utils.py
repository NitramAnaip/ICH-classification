import  numpy as np
from tensorflow.keras import backend as K
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


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))