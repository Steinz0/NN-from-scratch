from turtle import forward
import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        self._gradient = np.zeros(self._gradient.shape)

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass


class MSELoss(Loss):
    def forward(self, y, yhat):

        return (np.sum(y-yhat,axis=1)**2)

    def backward(self, y, yhat):
        return -2 * (y - yhat)

class Linear(Module):
    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output
        self._parameters = np.random.random((output,input.shape[1]))
    
    def forward(self, x):
        return x@self._parameters

    def backward_update_gradient(self, input, delta):
        pass        

x = np.random.random((50, 13))
y = np.random.random((50, 3))
yhat = np.random.random((50, 3))
y1 = np.array([[1,2,3],[5,6,7]])
y2 = np.array([[1,1,2],[5,6,7]])
mse = MSELoss()

loss = mse.forward(y,yhat)