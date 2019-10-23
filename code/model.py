
from data_loader import Dataloader
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.applications import mobilenet_v2
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

# def Mobile_NetV2():
# 	# create the base pre-trained model
# 	base_model = mobilenet_v2.MobileNetV2(include_top=True, weights='imagenet')


# 	x = base_model.output
# 	x = Dense(1024, activation='relu')(x)
# 	# and a logistic because we only have 3 classes
# 	predictions = Dense(3, activation='softmax')(x)

# 	model = Model(inputs=base_model.input, outputs=predictions)
# 	return model

# model=Mobile_NetV2()
# # first: train only the top layers (which were randomly initialized)
# # i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#     layer.trainable = False

# optimizer=optimizers.Adam(

# 	)
# # compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer=optimizer, loss='binary_crossentropy')


# # for i, layer in enumerate(base_model.layers):
# #    print(i, layer.name)
# # train the model on the new data for a few epochs
# #model.fit_generator(...)



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

	x = Dense(128, activation="relu")(x)
	x = Dense(128, activation="relu")(x)
	x = Dense(3, activation="sigmoid")(x)

	model = Model(inputs=inputs, outputs=x)

	return model


def my_2nd_model(input_shape):
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
	x = Conv2D(256, kernel_size=(3,3), strides=1, activation="relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(x)

	


	x=Flatten()(x)

	x = Dense(128, activation="relu")(x)
	x = Dense(128, activation="relu")(x)
	x = Dense(3, activation="sigmoid")(x)

	model = Model(inputs=inputs, outputs=x)

	return model




# optimizer=optimizers.Adam(

# 	)


optimizer=optimizers.SGD(
	lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True
	)
model=my_model(input_shape)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

nbr_epochs=300
generator=Dataloader(batch_size_train, percentage)


early_stopping = EarlyStopping(monitor="val_loss", patience=1, verbose=1, mode="min")
checkpoint_path=root+"results/checkpoints.hdf5"
checkpoint = ModelCheckpoint(checkpoint_path, period=1, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

a = model.fit_generator(
    generator.batch_yielder(), steps_per_epoch=203, epochs=nbr_epochs, callbacks=[early_stopping, checkpoint],
    validation_data=generator.validation_batch_yielder(), validation_steps=210
)



y=a.history['loss']
y_val=a.history['val_loss']
x=range(len(y_val))
plt.plot(x,y, y_val)
plt.savefig(root+"results/plots/plot_Adam.png")


model.load_weights(checkpoint_path) #loading the best weights to predict with these



test_paths=os.listdir(root+"data/test_images/test_images")
nbr_of_tests=len(test_paths)
probabilities=model.predict(generator.test_batch_yielder(), batch_size=1, verbose=0, steps=nbr_of_tests)

test_paths=np.array([test_paths]).T
probabilities=np.concatenate((test_paths, probabilities), axis=1)
print(probabilities)


def from_proba_to_output(probabilities):
	outputs=np.copy(probabilities)
	for i in range (len(outputs)):
		for j in range (1, 4):
			if (float(outputs[i][j]))>0.5:
				outputs[i][j]=1
			else:
				outputs[i][j]=0
	return outputs
outputs=from_proba_to_output(probabilities)
np.savetxt(root+"results/outputs_Adam.csv", outputs, fmt='%s', delimiter=",")
np.savetxt(root+"results/probabilities_Adam.csv", probabilities, fmt='%s', delimiter=",")




