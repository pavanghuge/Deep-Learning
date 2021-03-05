import sys

from keras import applications

from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D

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

input_size = Input(shape=(256, 256, 3))
transfer_model = applications.vgg16.VGG16(weights='imagenet', include_top = False, input_tensor = input_size, pooling = None )

for layer in transfer_model.layers:
  layer.trainable = False

output = transfer_model.output
output1 = GlobalAveragePooling2D()(output)
output2 = Dense(512, activation='relu')(output1)
output3 = Dense(128, activation='relu')(output2)
prediction = Dense(10, activation='softmax')(output3)

model = Model(inputs=transfer_model.input, outputs=prediction)

opt = optimizers.Adam(lr=0.001, beta_1=0.8)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

################# TRAINING MODEL AND SAVING MODEL #################
early_stopping = True
mc = ModelCheckpoint(model_file, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

def train():
    model.fit_generator(train_gen, 
                        steps_per_epoch=epoch_step, 
                        epochs=epochs,
                        validation_data=val_gen, 
                        validation_steps=val_steps, 
                        callbacks=[mc])
train()