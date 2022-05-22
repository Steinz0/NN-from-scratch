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
        return -np.sum(yhat*y, axis=1) + np.log(np.sum(np.exp(yhat), axis=1).reshape((-1,1)))
    def backward(self, y, yhat):
        """
        y -> One hot encoding
        """
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
        seuil = np.ones((yhat.shape[0], yhat.shape[1])) * -100
        return - (y*np.log(yhat + 1e-100) + (1-y)*np.log(1-yhat+ 1e-100))
        # return - (y * np.log(yhat + 1e-100) + (1-y) * np.log(1-yhat + 1e-100))
        return - (y * np.maximum(seuil, np.log(yhat)) + (1-y) * np.maximum(seuil, np.log(1-yhat)))
    def backward(self, y, yhat):
        """
        y -> One hot encoding
        """
        return -(y/(yhat + 1e-100)) + (1-y)/(1-yhat + 1e-100)
        seuil = np.ones((yhat.shape[0], yhat.shape[1])) * 1e-10000000000
        return -(y/np.maximum(seuil, yhat)) + (1-y)/np.maximum(seuil, 1-yhat)