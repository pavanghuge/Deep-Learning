
import sys
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils, generic_utils, to_categorical
from keras.layers import merge, Input
from keras.layers import Flatten
from keras.models import Model
from sklearn.utils import shuffle
from keras import optimizers
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

################# DEFINING FILE PATH AND HYPER-PARAMETERS #################

image_set = sys.argv[1]

model_file = sys.argv[2]

lr = 0.001
epochs = 5
val_steps = 8
epoch_step = 200

################# READING AND PROCESSING DATA #################

datagen = ImageDataGenerator(validation_split=0.2,rescale=1./255)

train_gen = datagen.flow_from_directory('{}'.format(image_set), subset='training', target_size=(256,256))

val_gen = datagen.flow_from_directory('{}'.format(image_set), subset='validation', target_size=(256,256))

################# DEVELOPING MODEL #################

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=[256,256,3], activation='relu'))#Convolution 1
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

#Adding Layers 
x = model.output
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)

# creating the model 
model = Model(input = model.input, output = predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, epsilon=1e-06),
              metrics=['accuracy'])

mc = ModelCheckpoint(model_file, monitor='val_acc', mode='max', verbose=1, save_best_only=True)


def train():
    model.fit_generator(train_gen,
              steps_per_epoch=epoch_step,
              epochs=epochs,
              validation_data = val_gen,
              validation_steps = val_steps,
              callbacks=[mc])
train()