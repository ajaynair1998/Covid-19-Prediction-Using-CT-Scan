import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, GlobalAveragePooling2D, ReLU, Dropout, Dense, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# relative path to data r'' is raw string
train_path = r'training_set'
validation_path = r'validation_set'

# functions for accessing directory path
__dirname = os.path.dirname(os.path.realpath(__file__))

# final directory
train_path = os.path.join(__dirname, train_path)
validation_path = os.path.join(__dirname, validation_path)

# Get images from both the classes using Image data generator
train_batches = ImageDataGenerator(rescale=1.0 / 255.0). \
    flow_from_directory(directory=train_path,
                        target_size=(512, 512),
                        color_mode='grayscale',
                        classes=['negative', 'positive'],
                        batch_size=8,
                        shuffle=True,
                        )

validation_batches = ImageDataGenerator(rescale=1.0 / 255.0). \
    flow_from_directory(directory=validation_path,
                        target_size=(512, 512),
                        color_mode='grayscale',
                        classes=['negative', 'positive'],
                        batch_size=4,
                        shuffle=True)

# for Debugging
train_batches_iterator = train_batches
print(train_batches.class_indices, train_batches_iterator.next())

# Defining Layers of the Model
model = Sequential([
    Conv2D(256, input_shape=(512, 512, 1), kernel_size=(2, 2),
           strides=(2, 2), padding='same', activation=tf.keras.activations.sigmoid),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
    Conv2D(512, strides=(1, 1), padding='same', kernel_size=(2, 2), input_shape=(256, 256, 1), activation=tf.keras.
           activations.sigmoid),
    ReLU(),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
    GlobalAveragePooling2D(),
    ReLU(),
    Dense(256),
    ReLU(),
    Dropout(0.5),
    Dense(2),
    Softmax(),

])

# Print out the Details of the model
model.summary()

# setting the optimiser and loss function
model.compile(optimizer=Adam(learning_rate=0.0004), loss='categorical_crossentropy', metrics=['accuracy'])

# training the model
model.fit(x=train_batches, validation_data=validation_batches, shuffle=True, epochs=10, verbose=1)

# Save the optimised weights after training
model.save_weights('/content/drive/MyDrive/tensorflow/saved_model/my_model.h5')
