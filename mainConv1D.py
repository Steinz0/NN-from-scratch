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

# dimensionX = len(alltrainx[0])
# number_neurons_layer1 = 100
# number_neurons_layer2 = 10
# # k_size, chan_in, chan_out, stride
# conv1D = Conv1D(1, 1, 2, 1)
# mp = MaxPool1D(2, 2)
# max_iter = 5000
# eps = 10e-5

x = np.array([[[1],[2],[3],[4]]])
print(x.shape)
# # x = np.array([alltrainx[0].reshape(-1, 1)])
# c1 = conv1D.forward(x)
# print(c1)
# input()
# p1 = mp.forward(c1)
# b = mp.backward_delta(c1, 2)
# print(p1)
# input()
# print(b)

#################### TEST ##################################
conv1D = Conv1D(3,1,32,1)
maxpool = MaxPool1D(2,2)
f = Flatten()
linear = Linear(4064, 100)
r = ReLu()
linear2 = Linear(100, 10)
mse = MSELoss()

alltrainx2 = np.array(alltrainx.reshape(len(alltrainx), -1, 1))

print(alltrainx2.shape)
seq = Sequential([conv1D, maxpool, f, linear, r, linear2])
ce = CrossEntropyLogSoftmax()

# yhat = seq.forward(alltrainx2[:2])
# c = conv1D.forward(alltrainx2[:2])
# c2 = maxpool.forward(c)
# c3 = f.forward(c2)
# c4 = linear.forward(c3)
# c5 = r.forward(c4)
# yhat = linear2.forward(c5)
# loss = mse.forward(alltrainy[:2], yhat)
# print(loss)

# bloss = mse.backward(alltrainy[:2], yhat)
# linear2.backward_update_gradient(c5, bloss)
# ld2 = linear2.backward_delta(c5, bloss)
# print(ld2.shape)
# br = r.backward_delta(c4, ld2)
# print(br.shape)
# linear.backward_update_gradient(c3, br)
# ld = linear.backward_delta(c3, br)
# print(ld.shape)
# bf = f.backward_delta(c2, ld)
# print(bf.shape)
# bp = maxpool.backward_delta(c, bf)
# print(bp.shape)
# conv1D.backward_update_gradient(alltrainx2[:2], bp)
# bc = conv1D.backward_delta(alltrainx2[:2], bp)
# print(bc.shape)


max_iter = 10
eps = 10e-5

all_loss = BatchGD(seq, ce, alltrainx2[:100], alltrainy[:100], epochs=max_iter, eps=eps, verbose=False)
print(alltrainy[0])
yhat = seq.forward(np.array([alltrainx2[0]]))
print(np.argmax(yhat)) 
