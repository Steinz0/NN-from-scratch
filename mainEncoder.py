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
number_neurons_layer1 = 100
number_neurons_layer2 = 10
print(dimensionX)
input()
seqEncoder = [Linear(dimensionX, number_neurons_layer1), TanH(), Linear(number_neurons_layer1, number_neurons_layer2), TanH()]
seqDecoder = [Linear(number_neurons_layer2, number_neurons_layer1), TanH(), Linear(number_neurons_layer1, dimensionX), Sigmoide()]

net = Sequential( seqEncoder + seqDecoder)
ce = BinaryCrossEntropy()

max_iter = 400
eps = 10e-5

all_loss = MiniBatchGD(net, ce, alltrainx, alltrainx, 1000, epochs=max_iter, eps=eps, verbose=False)

print(net.forward(alltrainx[0]))
plt.figure()
show_usps(alltrainx[0])
plt.show()

plt.figure()
show_usps(net.forward(alltrainx[0]))
plt.show()

plt.figure()
plt.plot(all_loss)
plt.show()
