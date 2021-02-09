import numpy as np
import cv2
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import random, os, glob
import matplotlib.pyplot as plt
batchsize = 64
train_path = 'mydata/training_set'
test_path = "mydata/test_set"
img_list = glob.glob(os.path.join(train_path, '*/*.jpg'))
len(img_list)
train = ImageDataGenerator(rescale=1./255,
                           horizontal_flip=True)
test = ImageDataGenerator(rescale=1./255,
                          horizontal_flip=True)
train_generator = train.flow_from_directory(train_path, target_size=(64, 64), batch_size=batchsize, class_mode = 'categorical')
test_generator = test.flow_from_directory(test_path, target_size=(64, 64), batch_size=batchsize, class_mode = 'categorical')
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
print(labels)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dropout(0.5)),
model.add(Dense(256, activation='relu'))
model.add(Dense(26, activation='softmax'))
filepath = "model.h5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit_generator(train_generator, epochs=100, steps_per_epoch=train_generator.samples//batchsize, validation_data=test_generator, validation_steps=test_generator.samples//batchsize,callbacks=callbacks_list, verbose=1)
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
