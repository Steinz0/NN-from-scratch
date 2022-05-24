import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass
    
class MSELoss(Loss):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y, yhat):
        return np.sum(y-yhat,axis=1)**2

    def backward(self, y, yhat):
        return -2*(y - yhat)

class CrossEntropyLogSoftmax(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        """
        y -> One hot encoding
        """
        # yhat = ((yhat.T/np.max(yhat, axis=1)).T/2)
        return -np.sum(yhat*y, axis=1) + np.log(np.sum(np.exp(yhat), axis=1).reshape((-1,1)))
        
    def backward(self, y, yhat):
        """
        y -> One hot encoding
        """
        # yhat = ((yhat.T/np.max(yhat, axis=1)).T/2)
        return -y + (np.exp(yhat))/np.sum(np.exp(yhat), axis=1).reshape((-1,1))

class CrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        """
        y -> One hot encoding
        """
        return -np.sum(y*yhat,axis=1)

    def backward(self, y, yhat):
        """
        y -> One hot encoding
        """
        return -1

class BinaryCrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        """
        y -> One hot encoding
        """
        return - (y*np.log(yhat + 1e-100) + (1-y)*np.log(1-yhat+ 1e-100))

    
    def backward(self, y, yhat):
        """
        y -> One hot encoding
        """
        
        return ((1-y)/(1-yhat+ 1e-100)) - (y/yhat+ 1e-100)