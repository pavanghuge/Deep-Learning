
# python Assignment4_SGD_MiniBatch.py ion.train.0.txt ion.test.0.txt 25
import numpy as np
#import sys

#################
### Read training data ###

trainfile = open("/Users/pavanghuge/Downloads/DeepLearning/MyAssignments/Assignment4/ion.train.0.txt")
#trainfile = open(sys.argv[1])
original_train_data = np.loadtxt(trainfile)



### Read testing data ###
testfile = open("/Users/pavanghuge/Downloads/DeepLearning/MyAssignments/Assignment4/ion.test.0.txt")
#testfile = open(sys.argv[2])
original_test_data = np.loadtxt(testfile)

# first column of labels
testlabels = original_test_data[:,0]

# rest of the columns are features
testData = original_test_data[:,1:]

# Adding Bias to test features
onearray = np.ones((testData.shape[0],1))
testData = np.append(testData,onearray,axis=1)

rows = original_train_data .shape[0]
cols = original_train_data .shape[1]

#miniBatchlen = int(sys.argv[3])
miniBatchlen = 25

hidden_nodes = 3

##############################
### Initialize all weights ###

# final layer weights
final_w = np.random.rand(hidden_nodes)

# hidden layer weights (nodes * no_features)
# since fully connected 
hidden_W = np.random.rand(cols, hidden_nodes)
epochs = 1000
eta = .001
prevObj = np.inf
currObj = 100000000
stop=0.000000001
iterations = 0

hidden_layer = np.zeros((miniBatchlen, hidden_nodes))
output_layer = np.zeros((miniBatchlen,))

###########################
### define a model to Calculate objective ###
def model(begin, end):
    global hidden_layer, output_layer
    hidden_layer = np.matmul(trainData[begin : end, ], hidden_W)
    sigmoid = lambda x: 1/(1+np.exp(-x))
    hidden_layer = np.array([sigmoid(x) for x in hidden_layer])
    output_layer = np.matmul(hidden_layer, final_w)
    currObj = np.sum(np.square(output_layer - trainLabels[begin : end, ]))
    return currObj
###############################
### Begin gradient descent ####
while(iterations < epochs):    
     #Update previous objective
    prevObj = currObj

    # shuffle data before updating gradient
    np.random.shuffle(original_train_data)

    # first column is labels
    trainLabels = original_train_data[:, 0]
        
    # REST OF THE coulmns are features 
    trainData = original_train_data[:, 1:]
        
    # adding bias to training data
    trainData = np.append(trainData, np.ones((trainData.shape[0],1)),axis = 1)


    num_batches = int(np.ceil(trainData.shape[0]/miniBatchlen))
    # gradient update for mini batch
    for j in range(0, num_batches):
            
        start_idx = j * miniBatchlen
        end_idx = start_idx + miniBatchlen if (start_idx + miniBatchlen) < trainData.shape[0] else trainData.shape[0]
        currObj = model(start_idx, end_idx)

        dellW = np.matmul((output_layer - trainLabels[start_idx : end_idx, ]) , hidden_layer)
        final_w = final_w - eta * dellW

        for i in range (0, hidden_nodes):
            dell_i = np.dot((output_layer - trainLabels[start_idx : end_idx, ]) * final_w[i] * (hidden_layer[:, i]) * (1-hidden_layer[:, i]), trainData[start_idx : end_idx, ])
            hidden_W [:,i] -= eta * dell_i 

    iterations = iterations + 1

    print("Epoch: %d, loss: " % iterations, currObj) 

#Predictions based on final weights
hidden_layer = np.matmul(testData, hidden_W)
sigmoid = lambda x: 1/(1 + np.exp(-x))
hidden_layer = np.array([sigmoid(x) for x in hidden_layer])
predictions = np.matmul(hidden_layer, np.transpose(final_w))
predictions = [-1 if prediction <= 0 else 1 for prediction in predictions]
        

print("\nPredictions  True Labels ") 
count = 0
for pred, trueLabels in zip(predictions,testlabels):
    if (pred != trueLabels):
        count +=1
    print("     ",pred, "          ", int(trueLabels) )
print("\nError Count: ",count)
print("\nTest Error: ",(count/ testlabels.shape[0])*100)
print("\nAccuracy: ",(1 - sum(abs(0.5*(testlabels - predictions)))/testlabels.shape[0]) * 100)
