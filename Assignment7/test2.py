# Test.py
import numpy as np
import os
from PIL import Image
from skimage.transform import resize
import sys

from keras.utils import np_utils, generic_utils, to_categorical
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras import backend as K
import keras
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.models import Model
from sklearn.utils import shuffle
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import optimizers
from keras.models import load_model

################# DEFINING FILE PATH AND HYPER-PARAMETERS ################# 

image_set = sys.argv[1]
model_file = sys.argv[2]

################# READING AND PROCESSING DATA #################

datagen = ImageDataGenerator(rescale=1./255)

test = datagen.flow_from_directory('{}'.format(image_set),
                                    target_size=(256,256))

y_test = test.classes


################ LOADING MODEL AND GENERATING OUTPUT #################

model = load_model(model_file)

loss, acc = model.evaluate_generator(test)

print("Loss:", loss)
print("Accuracy:", acc)
