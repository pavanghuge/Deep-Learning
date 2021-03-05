import sys
import numpy as np

trainFolder = sys.argv[1]
testFolder = sys.argv[2]

#################

data = open('{}/data.csv'.format(trainFolder)).readlines()
data = np.array([line.strip().split(',') for line in data[1:]])
train_img_file = data[:, 0]
trainLabel = data[:, 1].astype(np.float32)

data = open('{}/data.csv'.format(testFolder)).readlines()
data = np.array([line.strip().split(',') for line in data[1:]])
test_img_file = data[:, 0]
testLabel = data[:, 1].astype(np.float32)

train_imgs, test_imgs = [], []

for fl in train_img_file:
    train_imgs.append(np.loadtxt('{}/{}'.format(trainFolder, fl)))

trainData = np.array(train_imgs)
print("\nTrainData:\n",trainData)

for fl in test_img_file:
    test_imgs.append(np.loadtxt('{}/{}'.format(testFolder, fl)))

testData = np.array(test_imgs)

print("\nTestData:\n",testData)
print("\n\n")




################## 

eta = 0.01
prevObj = 100000001
currObjective = 100000000
iteration = 0
stop = 0
maxIter = 1000

weights = np.random.rand(2, 2)
print("c = \n",weights)

output_rows = trainData[1].shape[0] - weights.shape[0] + 1
output_cols = trainData[1].shape[1] - weights.shape[1] + 1
output = np.zeros((len(trainData), output_rows, output_cols))


def model():
    sigmoid = lambda x: 1/(1 + np.exp(-x))
    global weights, output

    for i, image in enumerate(trainData):
        for r in range(0, len(image)-1):
            for c in range(0, len(image[0])-1):
                im = image[r: r+weights.shape[0], c: c+weights.shape[1]]
                z = im * weights
                z = z.sum()
                output[i, r, c] = sigmoid(z)

    predicted = np.mean(output, axis=1).mean(axis=1)
    loss = sum(np.square(predicted - trainLabel[i]))
    return loss


currObjective = model()
while iteration < maxIter and (prevObj - currObjective) > stop:
    prevObj = currObjective
    dellW = np.zeros(weights.shape)

    for i, image in enumerate(trainData):
        z = np.array(output[i])
        loss = (z.sum() - trainLabel[i])
        for r in range(0, len(image)-1):
            for c in range(0, len(image[0])-1):
                im = np.array(image[r: r + weights.shape[0], c: c + weights.shape[1]])
                dot = z * (1 - z) * im
                dellW[r, c] += dot.sum() * loss

    weights -= eta * dellW
    currObjective = model()
    print("Epoch: {} Objective:{}".format(iteration, prevObj))
    iteration += 1


print("\n Kernel:\n ")
print(weights)

print("\n Predictions:\n")
testoutput = np.zeros((len(testData), output_rows, output_cols))
o = np.zeros(len(testData))
for i, image in enumerate(testData):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    for r in range(0, len(image) - 1):
        for c in range(0, len(image[0]) - 1):
            im = image[r: r + weights.shape[0], c: c + weights.shape[1]]
            z = im * weights
            z = z.sum()
            testoutput[i, r, c] = sigmoid(z)

Zmean = np.mean(testoutput, axis=1).mean(axis=1)
mean = testoutput.mean()
Zmean[Zmean >= mean] = 1
Zmean[Zmean < mean] = -1
for i, val in enumerate(testLabel):
    print(val, int(Zmean[i]))