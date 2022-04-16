from init_module import Loss
import numpy as np



class MSELoss(Loss):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y, yhat):
        return np.sum(y-yhat,axis=1)**2

    def backward(self, y, yhat):
        return -2*np.sum(y-yhat,axis=1)

def Softmax(ypred):
    return np.exp(ypred)/np.sum(np.exp(ypred))

def logSoftmax(ypred):
    return np.log(np.exp(ypred)/np.sum(np.exp(ypred)))

class CrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        return -yhat[y] + np.log(np.sum(np.exp(yhat[y])))

    def backward(self, y, yhat):
        return 
