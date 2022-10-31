from keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = "./Training_faces"
VALIDATION_DIR = "./Validation_faces"

train_datagen = ImageDataGenerator(rescale=1./255,                                      
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

import os
print(os.listdir(TRAINING_DIR))
print(os.listdir(VALIDATION_DIR))

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,target_size=(100,100),class_mode='categorical',color_mode='grayscale')
test_generator = test_datagen.flow_from_directory(VALIDATION_DIR, target_size=(100,100), class_mode='categorical',color_mode='grayscale')

from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import Sequential

model1 = Sequential()

model1.add(keras.Input(shape=(100, 100, 1)))

model1.add(layers.Conv2D(filters=64, kernel_size=(2,2), activation="relu"))
model1.add(layers.Dropout(0.2))
model1.add(layers.MaxPooling2D(pool_size=(2,2)))
model1.add(layers.Conv2D(filters=64, kernel_size=(2,2), activation="relu"))
model1.add(layers.Dropout(0.2))
model1.add(layers.MaxPooling2D(pool_size=(2,2)))
model1.add(layers.Dropout(0.2))
model1.add(layers.Conv2D(filters=128, kernel_size=(2,2), activation="relu"))

model1.add(layers.Flatten())

num_classes = 2
model1.add(layers.Dropout(0.2))
model1.add(layers.Dense(512,activation="relu"))
model1.add(layers.Dropout(0.2))
model1.add(layers.Dense(num_classes, activation="softmax"))
                                                                                                                                                                                                                                                                                                                                                                                                   
model1.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model1.fit(train_generator,validation_data=test_generator,epochs=12)

model1.save('grayscale_gender.h5')
