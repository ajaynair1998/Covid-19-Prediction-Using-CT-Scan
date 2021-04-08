import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, GlobalAveragePooling2D, ReLU, Dropout, Dense, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import os
# import shutil
# import glob
# import matplotlib.pyplot as plt
# import warnings
# from tensorflow.keras.metrics import categorical_crossentropy
# import itertools
# from tensorflow.keras import initializers
# from tensorflow.keras.layers import Activation, Flatten, InputLayer
# import numpy as np
# from sklearn.utils import shuffle
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import confusion_matrix
# from random import randint

# physical_devices=tf.config.experimental.list_physical_devices('GPU')
# print(len(physical_devices),physical_devices)


# relative path to data r'' is raw string
train_path = r'training_set'
validation_path = r'validation_set'

# functions for accessing directory path
__dirname = os.path.dirname(os.path.realpath(__file__))
# print(__dirname, '<------------->', os.path.realpath(__file__))
# print(os.path.join(__dirname, train_path_positive))

# initialisers for each layer
# layer1_bias=
# layer1_weights=


# final directory
train_path = os.path.join(__dirname, train_path)
validation_path = os.path.join(__dirname, validation_path)

# probably samplewise_std_normalization is zerocentre normalisation
train_batches = ImageDataGenerator(samplewise_std_normalization=True). \
    flow_from_directory(directory=train_path,
                        target_size=(512, 512),
                        color_mode='grayscale',
                        classes=['negative', 'positive'],
                        batch_size=10,
                        shuffle=True,
                        )

validation_batches = ImageDataGenerator(samplewise_std_normalization=True). \
    flow_from_directory(directory=validation_path,
                        target_size=(512, 512),
                        color_mode='grayscale',
                        classes=['negative', 'positive'],
                        batch_size=1,
                        shuffle=True)

train_batches_iterator = train_batches
print(train_batches.class_indices, train_batches_iterator.next())

# havent found out how to set initial bias yet
model = Sequential([
    Conv2D(512, input_shape=(512, 512, 1), kernel_size=(2, 2),
           strides=(2, 2), padding='same', activation=tf.keras.activations.sigmoid),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
    Conv2D(1024, strides=(1, 1), padding='same', kernel_size=(2, 2), activation=tf.keras.
           activations.sigmoid),
    ReLU(),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
    GlobalAveragePooling2D(),
    ReLU(),
    Dense(512),
    ReLU(),
    Dropout(0.5),
    Dense(2),
    Softmax(),

])

model.summary()

# setting the optimiser and loss function
model.compile(optimizer=Adam(learning_rate=0.0004), loss='categorical_crossentropy', metrics=['accuracy'])

# training the model
model.fit(x=train_batches, validation_data=validation_batches, shuffle=True, epochs=10, verbose=2)
