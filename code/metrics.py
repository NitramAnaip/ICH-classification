import numpy as np
from sampler import generator
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from utils import from_proba_to_output


class Metrics(Callback):
    def __init__(self, model, valid_data, network):
        super(Callback, self).__init__()
        self.val_f1=0
        self.val_recall=0
        self.val_precision=0
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.model=model
        self.valid_data=valid_data
        self.outputs_list=[]
        self.steps=len(valid_data)
        self.network=network


    def on_epoch_end(self, args,*kwargs):
        self.outputs_list=[]
        gen=generator(self.valid_data, self.network, self.outputs_list)
        val_predict = (np.asarray(self.model.predict(gen, batch_size=1, verbose=0, steps=self.steps)))
        val_predict=from_proba_to_output(val_predict, 0.5)
        self.val_f1 = f1_score(self.outputs_list, val_predict)
        self.val_recall = recall_score(self.outputs_list, val_predict)
        self.val_precision = precision_score(self.outputs_list, val_predict)
        self.val_f1s.append(self.val_f1)
        self.val_recalls.append(self.val_recall)
        self.val_precisions.append(self.val_precision)
        print ("val_f1: ", self.val_f1, "   val_precision: ", self.val_precision, "   _val_recall: ", self.val_recall)
        