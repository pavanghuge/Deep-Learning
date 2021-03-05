import sys
import numpy as np

from tensorflow.keras.models import load_model

################# DEFINING FILE PATH AND HYPER-PARAMETERS #################

data_file = sys.argv[1]
labels_file = sys.argv[2]

model_file = sys.argv[3]

################# READING AND PROCESSING DATA #################

x_test = np.load(data_file)
x_test = np.rollaxis(x_test, 1, 4)

y_test = np.load(labels_file)
y_test = np.reshape(y_test, (-1, 1))

################# LOADING MODEL AND GENERATING OUTPUT #################

model = load_model(model_file)

predictions = model.predict(x_test)
predictions = list(map(np.argmax, predictions))

count = 0

for pred, y_true in zip(predictions, y_test):
    
    if pred != y_true:
        count += 1

print("Test Error:", count/len(x_test))
