from turtle import forward
import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class Module(object):
    def __init__(self):
        self._w = None
        self._b = None
        self._gradient_w = None
        self._gradient_b = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._w -= gradient_step*self._gradient_w
        self._b -= gradient_step*self._gradient_b

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
        self._parameters = np.random.random((output,input))

    def forward(self, x):
        return x@self._w + self._b

    def backward_delta(self, input, delta):
        pass



