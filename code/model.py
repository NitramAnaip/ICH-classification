
from data_loader import Dataloader
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
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


root="/home/ubuntu/martin/Behold/"

# def VGG():
# 	model = applications.VGG16(include_top=False, weights='imagenet)

batch_size_train=30

percentage=0.75
batch_size_val=int(((1-percentage)/percentage)*batch_size_train)
input_shape=[128, 128]



#***************************************test part****************

def Mobile_NetV2():
	# create the base pre-trained model
	base_model = keras.applications.mobilenet.MobileNet(include_top=True, weights='imagenet')


	x = base_model.output
	x = Dense(1024, activation='relu')(x)
	# and a logistic because we only have 3 classes
	predictions = Dense(200, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions)
	return model


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

optimizer=optimizers.Adam(

	)
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=optimizer, loss='binary_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(...)



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
	x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(x)

	
	x = Conv2D(256, kernel_size=(3,3), strides=1, activation="relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(x)

	


	x=Flatten()(x)

	#x = Dense(4096, activation="relu")(x)
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

nbr_epochs=300
train=Dataloader(batch_size_train, True, percentage)
valid=Dataloader(batch_size_val, False, percentage) 

early_stopping = EarlyStopping(monitor="val_loss", patience=50, verbose=1, mode="min")
checkpoint = ModelCheckpoint(root+"results/checkpoints.hdf5", period=1, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

a = model.fit_generator(
    train.batch_yielder(), steps_per_epoch=210, epochs=nbr_epochs, callbacks=[early_stopping, checkpoint],
    validation_data=valid.batch_yielder(), validation_steps=210
)



y=a.history['loss']
y_val=a.history['val_loss']
x=range(len(y_val))
plt.plot(x,y, y_val)
plt.savefig(root+"results/plots/plot_Adam.png")

test_paths=os.listdir(root+"data/test_images/test_images")
nbr_of_tests=len(test_paths)
test=Dataloader(1, False, percentage) 
results=model.predict(valid.test_batch_yielder(), batch_size=1, verbose=0, steps=nbr_of_tests)

test_paths=np.array([test_paths]).T
print(results)
results=np.concatenate((test_paths, results), axis=1)
print(results)


np.savetxt(root+"results/results_Adam.csv", results, fmt='%s', delimiter=",")



