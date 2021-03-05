import sys
import numpy as np

from sklearn.metrics import accuracy_score

from keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model

################# DEFINING FILE PATH AND HYPER-PARAMETERS #################

data_file = sys.argv[1]

model_file = sys.argv[2]

################# READING AND PROCESSING DATA #################

datagen = ImageDataGenerator(rescale=1./255)

test = datagen.flow_from_directory('{}'.format(data_file),
                                    target_size=(256,256),
				    batch_size=1)

y_test = test.classes

################# LOADING MODEL AND GENERATING OUTPUT #################

model = load_model(model_file)

loss, acc = model.evaluate_generator(test)

print("Loss:", loss)
print("Accuracy:", acc)
