# MaskCheck-CNN
This program employs a Convolutional Neural Network , in order to both implement a face detection system which in its first iteration can detection the distinction between male and female faces , and in the second iteration we use a different model which trained through the same base code to detect masks on faces , in a live setting.


## Acknowledgements

 - [Kaggle Datasets](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
 - [Research Paper](https://ieeexplore.ieee.org/document/9342585)
## Steps
1. Take Photographic datasets from the web
2. Label the Datasets 
3. Split the dataset into Validation and Training set.
4. Convert the Datasets and normalize them into arrays
5. Construct the Convolutional Neural Network.
6. Train the Convolutional Nueral network.
7. Save the model with 'h5' extension.
8. Load in the model file through tensorflow .
9. Construct OpenCv based pipeline to record and register faces .
10. Convert those detected faces to grayscale.
11. Feed the captured grayscale images to the model in each instance.
12. Set up the labels for the probablities.
13. Run the Implementation.

## Code
Importing keras
```
import keras
```
Setup up directory variables and Validation sets as well create label generators with keras ImageDataGenerator.
```
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
```
Load in Data
```
import os
print(os.listdir(TRAINING_DIR))
print(os.listdir(VALIDATION_DIR))
----------------------------------------------------------
['unmasked', 'masked']
['unmasked', 'masked']
```
Run the DATAGEN to label training sets
```
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,target_size=(100,100),class_mode='categorical',color_mode='grayscale')
----------------------------------------------------------
Found 1207 images belonging to 2 classes.
```
Run the DATAGEN to lable test sets
```
test_generator = test_datagen.flow_from_directory(VALIDATION_DIR, target_size=(100,100), class_mode='categorical',color_mode='grayscale')
----------------------------------------------------------
Found 169 images belonging to 2 classes.
```
Importing Libraries
```
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import Sequential
```
Constructing the model
```
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
-----------------------------------------------------------------
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 99, 99, 64)        320       
                                                                 
 dropout (Dropout)           (None, 99, 99, 64)        0         
                                                                 
 max_pooling2d (MaxPooling2D  (None, 49, 49, 64)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 48, 48, 64)        16448     
                                                                 
 dropout_1 (Dropout)         (None, 48, 48, 64)        0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 24, 24, 64)       0         
 2D)                                                             
                                                                 
 dropout_2 (Dropout)         (None, 24, 24, 64)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 23, 23, 128)       32896     
                                                                 
 flatten (Flatten)           (None, 67712)             0         
                                                                 
 dropout_3 (Dropout)         (None, 67712)             0         
                                                                 
 dense (Dense)               (None, 512)               34669056  
                                                                 
 dropout_4 (Dropout)         (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 2)                 1026      
                                                                 
=================================================================
Total params: 34,719,746
Trainable params: 34,719,746
Non-trainable params: 0
_________________________________________________________________

```
train the model
```
model1.fit(train_generator,validation_data=test_generator,epochs=12)
-----------------------------------------------------------------------------
Epoch 1/12
1470/1470 [==============================] - 367s 244ms/step - loss: 0.6339 - accuracy: 0.6441 - val_loss: 0.5177 - val_accuracy: 0.7593
Epoch 2/12
1470/1470 [==============================] - 84s 57ms/step - loss: 0.5357 - accuracy: 0.7352 - val_loss: 0.3813 - val_accuracy: 0.8425
Epoch 3/12
1470/1470 [==============================] - 83s 57ms/step - loss: 0.4281 - accuracy: 0.8045 - val_loss: 0.2981 - val_accuracy: 0.8850
Epoch 4/12
1470/1470 [==============================] - 83s 57ms/step - loss: 0.3581 - accuracy: 0.8441 - val_loss: 0.2368 - val_accuracy: 0.9115
Epoch 5/12
1470/1470 [==============================] - 83s 57ms/step - loss: 0.3209 - accuracy: 0.8644 - val_loss: 0.2171 - val_accuracy: 0.9143
Epoch 6/12
1470/1470 [==============================] - 83s 56ms/step - loss: 0.3058 - accuracy: 0.8734 - val_loss: 0.2236 - val_accuracy: 0.9182
Epoch 7/12
1470/1470 [==============================] - 84s 57ms/step - loss: 0.2853 - accuracy: 0.8817 - val_loss: 0.1997 - val_accuracy: 0.9223
Epoch 8/12
1470/1470 [==============================] - 86s 58ms/step - loss: 0.2759 - accuracy: 0.8877 - val_loss: 0.2070 - val_accuracy: 0.9178
Epoch 9/12
1470/1470 [==============================] - 87s 59ms/step - loss: 0.2679 - accuracy: 0.8916 - val_loss: 0.1756 - val_accuracy: 0.9388
Epoch 10/12
1470/1470 [==============================] - 86s 59ms/step - loss: 0.2571 - accuracy: 0.8969 - val_loss: 0.1922 - val_accuracy: 0.9368
Epoch 11/12
1470/1470 [==============================] - 86s 58ms/step - loss: 0.2551 - accuracy: 0.8967 - val_loss: 0.2096 - val_accuracy: 0.9265
Epoch 12/12
1470/1470 [==============================] - 86s 58ms/step - loss: 0.2501 - accuracy: 0.9003 - val_loss: 0.1794 - val_accuracy: 0.9369

```
Saving the Model
```
model1.save('mask_model.h5')
```
Running the OpenCv Model Implementation.

```
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
                    
# load model
model = load_model('./grayscale_gender.h5')

# open webcam
webcam = cv2.VideoCapture(0)
    
classes = ['woman','man']

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)


    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.cvtColor(face_crop,cv2.COLOR_BGR2GRAY)
        face_crop = face_crop.astype("float") / 255.0
        face_crop = cv2.resize(face_crop,(100,100))
        face_crop = np.reshape(face_crop,[1,100,100,1])

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()

```
Test Photographs - Masked and Unmasked.
![test_img](https://user-images.githubusercontent.com/111745916/206692740-8a1693fd-6db4-4193-becf-1bf4b16bd785.png)
![second_img](https://user-images.githubusercontent.com/111745916/206692750-3767f7f5-087e-46c6-a598-e16fb9cf9399.png)

