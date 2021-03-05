import sys
import numpy as np

import keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D

from keras.callbacks import EarlyStopping, ModelCheckpoint

################# DEFINING FILE PATH AND HYPER-PARAMETERS #################

data_file = sys.argv[1]
labels_file = sys.argv[2]

model_file = sys.argv[3]

num_labels = 10
lr = 0.001
epochs = 50
batch_size = 32

################# READING AND PROCESSING DATA #################

x_train = np.load(data_file)
x_train = np.rollaxis(x_train, 1, 4)


y_train = np.load(labels_file)
y_train = np.reshape(y_train, (-1, 1))
y_train = keras.utils.to_categorical(y_train, num_labels)

train_data, train_labels = x_train[:12000], y_train[:12000]
test_data, test_labels = x_train[12000:], y_train[12000:]

################# DEVELOPING MODEL #################

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=[112,112,3], activation='relu'))#Convolution 1
model.add(BatchNormalization(momentum=0.6))
model.add(Dropout(0.4))

model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))#Convolution 2
model.add(BatchNormalization(momentum=0.4))
model.add(Dropout(0.4))

model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))#Convolution 3
model.add(BatchNormalization(momentum=0.4))
model.add(Dropout(0.4))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))#Convolution 4
model.add(BatchNormalization(momentum=0.4))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.4))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, epsilon=1e-06)


model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

################# TRAINING MODEL AND SAVING MODEL #################

early_stopping = True

if early_stopping:

  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
  mc = ModelCheckpoint(model_file, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

  def train():
      model.fit(train_data, train_labels,
                batch_size=batch_size,
                epochs=50,
                validation_data=(test_data, test_labels),
                callbacks=[es, mc],
                shuffle=True)

else:

  def train():
      model.fit(train_data, train_labels,
                batch_size=batch_size,
                epochs=12,
                validation_data=(test_data, test_labels),
                shuffle=True)

train()
