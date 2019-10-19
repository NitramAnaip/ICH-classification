
from data_loader import Dataloader
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import optimizers, losses
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Dense,
    Conv2D,
    Conv3D,
    Flatten,
    MaxPooling2D,
    Input,
    Lambda
    )

# def VGG():
# 	model = applications.VGG16(include_top=False, weights='imagenet)

batch_size_train=30

percentage=0.75
batch_size_val=int(((1-percentage)/percentage)*batch_size_train)
input_shape=[128, 128]


def my_model(input_shape):
	inputs = Input(shape=(input_shape[1], input_shape[0],1))

	x = Lambda(lambda i: (tf.to_float(i) / 255))(inputs)

	x = Conv2D(128, kernel_size=(5, 5), strides=2, activation="relu")(inputs)
	x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(x)

	x = Conv2D(128, kernel_size=(3,3), strides=1, activation="relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(x)

	x = Conv2D(128, kernel_size=(3,3), strides=1, activation="relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(x)

	x = Conv2D(128, kernel_size=(3,3), strides=1, activation="relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(x)

	x=Flatten()(x)

	x = Dense(64, activation="relu")(x)
	x = Dense(3, activation="softmax")(x)

	model = Model(inputs=inputs, outputs=x)

	return model

optimizer=optimizers.Adam(

	)
# optimizer=optimizers.SGD(
# 	lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True
# 	)
model=my_model(input_shape)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

nbr_epochs=15
train=Dataloader(batch_size_train, True, percentage)
valid=Dataloader(batch_size_val, False, percentage)
a = model.fit_generator(
    train.batch_yielder(), steps_per_epoch=210, epochs=nbr_epochs, 
    validation_data=valid.batch_yielder(), validation_steps=210
)


import pdb
pdb.set_trace()

y=a.history['loss']
y_val=a.history['val_loss']
x=range(nbr_epochs)
plt.plot(x,y, y_val)
plt.savefig("/home/ubuntu/Stages/exos_stages/Behold/plots/plot.png")
