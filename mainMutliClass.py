import numpy as np

from classes import *
from loss import *
import numpy as np
from mltools import *
from tqdm import tqdm
import matplotlib.pyplot as plt

def one_hot(Y):
    uniqueValue = np.unique(Y)
    one_hot_Y = np.zeros((len(Y), len(uniqueValue)))

    for i in range(len(Y)):
        one_hot_Y[i][Y[i]] = 1

    return one_hot_Y

############### USPS DATA ##########################################
uspsdatatrain = "./data/USPS_train.txt"
uspsdatatest = "./data/USPS_test.txt"
alltrainx,alltrainy = load_usps(uspsdatatrain[:])
alltestx,alltesty = load_usps(uspsdatatest[:])

alltrainy = one_hot(alltrainy)
alltesty = one_hot(alltesty)
############## INITIALIZE NEURAL NETWORK ##########################################

dimensionX = len(alltrainx[0])
number_neurons_layer1 = 256
number_neurons_layer2 = 128
number_neurons_layer3 = 64
number_output = 10

seqCE = Sequential([Linear(dimensionX,number_neurons_layer1),TanH(),Linear(number_neurons_layer1,number_neurons_layer2),TanH(),Linear(number_neurons_layer2,number_neurons_layer3),TanH(),Linear(number_neurons_layer3,number_output)])
ce = CrossEntropyLogSoftmax()

# mean, std = SGD(X_train,y_train,batchsize,iteration,earlystop=50)
# plt.figure()
# plt.plot(mean)
# plt.plot(std)
# plt.legend(('Mean', 'std'))
# plt.show()

max_iter = 0
eps = 10e-4
opt = Optim(seqCE,ce,eps=eps)
all_loss = SGD(seqCE, ce, alltrainx, alltrainy, 1000, max_iter=1000, eps=eps, verbose=True)
# all_loss_train, all_loss_test = SGDWithTest(seqCE, ce, alltrainx, alltrainy, alltestx, alltesty, 1000, max_iter=1000, eps=eps, verbose=False)
plt.figure()
plt.plot(all_loss)
# plt.plot(all_loss_test)
# plt.legend(('Train', 'Test'))
plt.show()

def accuracy(x, y, net):
    yhat = net.forward(x)
    preds = np.argmax(yhat, axis=1)
    y = np.argmax(y, axis=1)
    
    return np.where(preds == y, 1, 0).sum()/len(y)

print(accuracy(alltestx, alltesty, seqCE))