from classes import *
from loss import *
import numpy as np
from mltools import *
from tqdm import tqdm
from keras.datasets import mnist

############## RANDOM DATA #####################################

trainx = np.random.random((1000,256))
trainy = np.random.random((1000,2))

############### USPS DATA ######################################
uspsdatatrain = "./data/USPS_train.txt"
uspsdatatest = "./data/USPS_test.txt"
alltrainx,alltrainy = load_usps(uspsdatatrain)
alltestx,alltesty = load_usps(uspsdatatest)
neg = 5
pos = 6
X_train_usps,Y_train_usps = get_usps([neg,pos],alltrainx,alltrainy)
X_test_usps,Y_test_usps = get_usps([neg,pos],alltestx,alltesty)

############### MNIST ######################################

(X_mnist_train, y_mnist_train), (X_mnist_test, y_mnist_test) = mnist.load_data()
X_mnist_train = X_mnist_train.reshape(60000, 784)
X_mnist_test = X_mnist_test.reshape(10000, 784)

X_mnist_train = X_mnist_train[:2000]
y_mnist_train = y_mnist_train[:2000]
X_mnist_test = X_mnist_test[:1000]
y_mnist_test = y_mnist_test[:1000]

############### 4 GAUSIENNES DATA ######################################

artix, artiy = gen_arti(data_type=1)
artiy = np.where(artiy == -1, 0, 1)

artix_train, artiy_train, artix_test, artiy_test = [], [], [] ,[]

part = 5 #1/part of the data for test

for i in range(len(artix)):
    if i%part == 0:
        artix_test.append(artix[i])
        artiy_test.append(artiy[i])
    artix_train.append(artix[i])
    artiy_train.append(artiy[i])

artix_train, artiy_train, artix_test, artiy_test = np.array(artix_train), np.array(artiy_train), \
                                                   np.array(artix_test), np.array(artiy_test)
################### INITIALISATION NN ##################################

l1 = Linear(len(artix[0]),4)
t1 = TanH()
l2 = Linear(4,1)
t2 = TanH()
l3 = Linear(4,1)
s = Sigmoide()

seq = Sequential([l1,t1,l2,s])

seqCE = Sequential([Linear(len(alltrainx[0]),256),TanH(),Linear(256,128),TanH(),Linear(128,10)])
seqMNIST = Sequential([Linear(len(X_mnist_train[0]),128),TanH(),Linear(128,64),TanH(),Linear(64,10)])

mse = MSELoss()

ce = CrossEntropy()
cels = CrossEntropyLogSoftmax()

seqEncoder = Sequential([Linear(256,100), TanH(), Linear(100,10), TanH(), Linear(10,100), TanH(), Linear(100,256), Sigmoide()])

################### UTILS FUNCTION ##################################

def predict(X,seq):
    res = np.where(seq.forward(X) > 0.5, 1, 0)
    return res

def predict_argmax(X,seq):
    res = np.argmax(seq.forward(X), axis=1)
    return res

def plot_decision(X, Y, pred_function, seq):
    plt.figure()
    plot_frontiere(X, pred_function, seq)
    plot_data(X, Y)
    plt.show()

def accuracy(X, Y, seq, verbose=False):
    pred = seq.forward(X)
    accu = 0
    for i in range(len(Y)):
        if verbose:
            if i % 10 == 0:
                input()
        if pred[i] == Y[i]:
            accu +=1
    return accu/len(Y)
    
def one_hot(Y):
    uniqueValue = np.unique(Y)
    one_hot_Y = np.zeros((len(Y), len(uniqueValue)))

    for i in range(len(Y)):
        one_hot_Y[i][Y[i]] = 1

    return one_hot_Y

################### MAIN ##################################

if __name__ == '__main__':
    max_iter = 1000
    eps = 10e-7
    # SGD(seqCE, mse, alltestx, alltesty.reshape((-1,1)), 1000, max_iter=max_iter, eps=eps, verbose=True)

    y_mnist_train_one_hot = one_hot(y_mnist_train)
    for i in range(max_iter):
        yhat = seqMNIST.forward(X_mnist_train)
        loss = mse.forward(y_mnist_train_one_hot, yhat)
        print(f'Iteration {i} : loss {loss.mean()}')
        bloss = mse.backward(y_mnist_train_one_hot, yhat)
        seqMNIST.backward(bloss)
        seqMNIST.update_parameters(eps)

    # plot_decision(artix_train, artiy_train, predict_argmax, seqCE)
    print(accuracy(X_mnist_train, y_mnist_train_one_hot, seqMNIST))