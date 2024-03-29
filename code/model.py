from data_loader import Dataloader
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from sampler import Sampler, DataLoader, test_generator
from utils import from_proba_to_output, tranf_for_conf_matrix
from metrics import Metrics
from sklearn.metrics import roc_curve, confusion_matrix
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import optimizers, losses
from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.keras.models import Model
from tensorflow.keras.metrics import AUC, Recall, Precision
from tensorflow.python.keras.layers import (
    Dense,
    Conv2D,
    Conv3D,
    Flatten,
    MaxPooling2D,
    Input,
    Lambda
    )





root = "/home/ubuntu/martin/kaggle/"


batch_size_train = 30

percentage = 0.75
batch_size_val = int(((1 - percentage) / percentage) * batch_size_train)
input_shape = [128, 128]

nbr_epochs = 300

mobilenet=False


network="my_2nd_model"
opti="SGD"
if network=="mobilenet":
    mobilenet=True

# generator=DataLoader(batch_size_train, percentage, mobilenet






early_stopping = EarlyStopping(monitor="val_acc", patience=2, verbose=1, mode="max")
checkpoint_path=root+"results/checkpoints.hdf5"
checkpoint = ModelCheckpoint(checkpoint_path, period=1, monitor="val_acc", verbose=1, save_best_only=True, mode="max")




#***************************************Mobilenet part****************


if network=="mobilenet":
    base_model = mobilenet_v2.MobileNetV2(input_shape=(128, 128, 3), include_top=True, weights='imagenet')



    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    # and a logistic because we only have 3 classes
    predictions = Dense(3, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    optimizer=optimizers.Adam(

        )
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')



    #train the model on the new data for a few epochs
    model.fit_generator(
        generator.batch_yielder(),
        epochs=10,
        steps_per_epoch=203,
        verbose=1,
        validation_data=generator.validation_batch_yielder(),
        validation_steps=203,
        max_queue_size=10
    )

    #Fine tuning the mobilenet layers

    for layer in model.layers:
       layer.trainable = True

#*********************************************

def my_model(input_shape):
    inputs = Input(shape=(input_shape[1], input_shape[0],1))

    x = Lambda(lambda i: (tf.to_float(i) / 255))(inputs)

    x = Conv2D(64, kernel_size=(3,3), strides=1, activation="relu")(inputs)
    x = Conv2D(64, kernel_size=(3,3), strides=1, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(x)

    x = Conv2D(128, kernel_size=(3,3), strides=1, activation="relu")(x)
    x = Conv2D(128, kernel_size=(3,3), strides=1, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(x)

    x = Conv2D(128, kernel_size=(3,3), strides=1, activation="relu")(x)
    x = Conv2D(128, kernel_size=(3,3), strides=1, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(x)

    
    x = Conv2D(256, kernel_size=(3,3), strides=1, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(x)

    


    x=Flatten()(x)

    #x = Dense(4096, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)

    return model




def my_2nd_model(input_shape):
    inputs = Input(shape=(input_shape[1], input_shape[0], 1))

    x = Lambda(lambda i: (tf.to_float(i) / 255))(inputs)

    x = Conv2D(64, kernel_size=(3, 3), strides=1, activation="relu")(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=1, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(
        x
    )

    x = Conv2D(128, kernel_size=(3, 3), strides=1, activation="relu")(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=1, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(
        x
    )

    x = Conv2D(256, kernel_size=(3, 3), strides=1, activation="relu")(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(
        x
    )

    x = Conv2D(512, kernel_size=(3, 3), strides=1, activation="relu")(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=1, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(
        x
    )

    x = Flatten()(x)

    x = Dense(128, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)

    return model





if network=="my_model":
    model=my_model(input_shape)
elif network=="my_2nd_model":
    model=my_2nd_model(input_shape)

if opti=="Adam":
    optimizer=optimizers.Adam(

    )
if opti=="SGD":
    optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)



model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])
model.summary()

nbr_epochs=300




# a = model.fit_generator(
#     generator.batch_yielder(), steps_per_epoch=210, epochs=nbr_epochs, callbacks=[early_stopping, checkpoint],
#     validation_data=generator.validation_batch_yielder(), validation_steps=210
# )


#NEW TRYOUT**********************
partition=[0.75,0.15,0.1]
data=DataLoader(0,partition)
data.balance_data()
train_data=data.train_data
valid_data=data.valid_data
test_data=data.test_data
nbr_pos_in_train=data.nbr_pos_in_train

# import pdb
# pdb.set_trace()
train_sampler=Sampler(train_data, batch_size_train, "train", network, nbr_pos_in_train)
valid_sampler=Sampler(valid_data, batch_size_train, "valid", network)
a = model.fit_generator(
    train_sampler, epochs=6, verbose=1, validation_data=valid_sampler, max_queue_size=1,
    workers=1, use_multiprocessing=True, callbacks=[early_stopping, checkpoint, Metrics(model, valid_data, network)]
)
#*********************








# ploting for better understanding of what's happening

y=a.history['loss']
y_val=a.history['val_loss']
x=range(len(y_val))
plt.plot(x,y, y_val)
plt.savefig(root + "plots/plot_"+opti+"_"+network+".png")


test_gen=test_generator(test_data, network)

# Predicting and saving results

model.load_weights(checkpoint_path)  # loading the best weights to predict with these


# nbr_of_tests = len(test_paths)
probabilities = model.predict(
    test_gen, batch_size=1, verbose=0, steps=len(test_data)
)
# test_paths = np.array([test_paths]).T
# probabilities = np.concatenate((test_paths, probabilities), axis=1)
# print(probabilities)


def from_proba_to_output(probabilities, threshold):
    outputs = np.copy(probabilities)
    for i in range(len(outputs)):
        for j in range(2):
            if (float(outputs[i][j])) > threshold:
                outputs[i][j] = 1
            else:
                outputs[i][j] = 0
    return outputs

# def transf_for_roc(probabilities):
#     outputs=[]
#     for i in range(len(probabilities)):
#         outputs.append(probabilities[i][1])#we output the probability of the img being one of a hemorragy
#     return outputs



# outputs = from_proba_to_output(probabilities)
# np.savetxt(root + "results/outputs/"+opti+"_"+network+".csv", outputs, fmt="%s", delimiter=",")
# np.savetxt(
#     root + "results/probabilities/"+opti+"_"+network+".csv", probabilities, fmt="%s", delimiter=","
# )



#ROC curve plotting
y_score=probabilities
y_true=[]
for i in range (len(test_data)):
    y_true.append(test_data[i][1]) #if in a state of hemorragy it will be of value 1
y_score, y_true=np.array(y_score), np.array(y_true)
print("y_true: ", y_true)


fpr, tpr, thresholds = roc_curve(y_true, y_score)
print(fpr, tpr, thresholds)
plt.clf()
plt.plot(fpr,tpr)
plt.ylabel('tpr')
plt.xlabel('fpr')
plt.savefig("/home/ubuntu/martin/kaggle/plots/roc.png")

plt.clf()
plt.plot(thresholds,tpr, label='tpr')
plt.plot(thresholds,fpr, label='fpr')
plt.xlabel('thresholds')
plt.savefig("/home/ubuntu/martin/kaggle/plots/thresholds.png")

y_pred=tranf_for_conf_matrix(probabilities, 0.5)
print(y_pred)
print(confusion_matrix(y_true, y_pred))

# x=range (0, 9)
# y=range (0,9)
# plt.plot(x,y)
# plt.savefig("/home/ubuntu/martin/kaggle/plots/test.png")

#metrics to use with patient detection: Recall (TPR) or AUC