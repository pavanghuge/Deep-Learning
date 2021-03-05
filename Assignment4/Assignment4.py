
# python Assignment4.py ion.train.0.txt ion.test.0.txt 25
import numpy as np
import sys 

######### reading training file #######
#train_file = sys.argv[1]
#train = open(train_file)
train= open("/Users/pavanghuge/Downloads/DeepLearning/MyAssignments/Assignment4/ion.train.0.txt")
traindata = np.loadtxt(train)
one_arr = np.ones((traindata.shape[0],1))
train_data = np.append(one_arr, traindata, axis = 1)
train_data = train_data[ : , 1: ]
trainlabels = train_data[:,0]



######### reading testing file #######
#test_file = sys.argv[2]
#test = open(test_file)
test = open("/Users/pavanghuge/Downloads/DeepLearning/MyAssignments/Assignment4/ion.test.0.txt")
testdata =np.loadtxt(test)
one_arr = np.ones((testdata.shape[0],1))
test_data = np.append(one_arr,testdata,axis = 1)
test_data = test_data[ : , 1: ]
testlabels = test_data[:,0]


####### defining parameters #########
# batch size
k = 25 #int(sys.argv[3])

hidden_nodes = 3
eta =0.001
theta = 0.000000001
epochs = 1000


# function to calculate loss function
def Loss(data,labels,weights):
    objective = np.matmul(data,np.transpose(weights)) -labels 
    objective = np.sum(np.square(objective))
    return objective

# function to calculate sigmoid 
def sigmoid (x):
    sig = 1 + np.exp(-x)
    return 1/sig

#function to calculate derivative of sigmoid 
def sig_d(x):
    return x * (1 - x)

#function to calculate delF of output layer.
def delfW_func(data,labels, weights):
    # initializing to zeros
    delfW = np.zeros((data.shape[1]))

    for i, row in enumerate(data):
        delfW += (np.dot(row, np.transpose(weights))- labels[i]) * row
    
    return delfW

#function to calculate delF for hidden layers
def delfH_func(data, hidden_layer,labels,weights):

    # hidden layer dimension 
    dim_hidden = data.shape[1]

    # initializing to zeros
    delfH = np.zeros((hidden_nodes, dim_hidden)) 

    for i in range(hidden_nodes):
        for j, row in enumerate(data):
            error = np.dot(hidden_layer[j],np.transpose(weights))-labels[j]
            sig_diff =  sig_d(hidden_layer[j][i])

            delfH[i] += error * weights[i] * sig_diff * row
    
    return delfH



# trainig model
def train_model(train_data, trainlabels,test_data,testlabels):

    dim_h = len(train_data[0])
    final_weights = np.random.rand(hidden_nodes)
    hidden_weights = np.random.rand( hidden_nodes, dim_h)

    for i in range(epochs):
        np.random.shuffle(train_data)
        train_features =  train_data
        train_labels = trainlabels
        no_batches = int(np.ceil(len(train_data)/k))

        for j in range(no_batches):
            start = j
            end = j + k if j + k < len(train_data) else len(train_data) 

            mini_features = np.array(train_features[start:end])
            mini_labels = np.array(train_labels[start:end])

            hidden_layer = np.matmul(mini_features, np.transpose(hidden_weights))
            hidden_layer = np.array([sigmoid(node) for node in hidden_layer])

            delfW = delfW_func(hidden_layer, mini_labels, final_weights)
            final_weights -= eta * delfW

            delfH = delfH_func(mini_features,hidden_layer,mini_labels,final_weights)
            hidden_weights -= eta * delfH
        
        print("Epoch: ",i, "  Objective: ",Loss(hidden_layer,mini_labels,final_weights))
       
    
    hidden_layer1 = np.matmul(test_data, np.transpose(hidden_weights))
    pred1 = np.array([sigmoid(node) for node in hidden_layer1])
    output_layer1 = np.dot(pred1, np.transpose(final_weights))
    pred = np.sign(output_layer1)
    count = 0
    print("\nPredictions\t Truth\n")
    for i, predictions in enumerate(pred):
        print(predictions, "\t", testlabels[i])
        if predictions != testlabels[i]:
            count += 1
    
    print("Error count: ", count)
    print ("Error rate: ", count/len(testlabels))
    print("\nAccuracy: ",(1 - sum(abs(0.5*(testlabels - predictions)))/testlabels.shape[0]) * 100,"\n\n\n")




train_model(train_data,trainlabels, test_data,testlabels)









