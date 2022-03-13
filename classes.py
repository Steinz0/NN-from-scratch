from ast import Mod
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
        pass

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
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y, yhat):
        return np.sum(y-yhat,axis=1)**2

    def backward(self, y, yhat):
        return 2*np.sum(np.abs(y-yhat),axis=1)

class Linear(Module):
    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output
        self._parameters = np.random.random((input, output))
        self._gradient = np.zeros((input, output))
    
    def forward(self, X):
        return np.dot(X,self._parameters)

    def zero_grad(self):
        self._gradient = np.zeros((self.input,self.output))

    def backward_update_gradient(self, input, delta):
        self._gradient += input.T @ delta.reshape((delta.shape[0],-1))

    def backward_delta(self, input, delta):
        return input.T @ delta

    def update_parameters(self, gradient_step=0.001):
        self._parameters -= gradient_step * self._gradient


