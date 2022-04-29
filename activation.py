from init_module import Module
import numpy as np

class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return ( 1 - np.tanh(input)**2 ) * delta

    def update_parameters(self, gradient_step=0.001):
        pass

class Sigmoide(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return 1/(1+np.exp(-X))

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        delta = delta.reshape((delta.shape[0],-1))
        f = 1/(1+np.exp(-input))
        return (f * (1- f)) * delta

    def update_parameters(self, gradient_step=0.001):
        pass

class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.exp(X)/np.sum(np.exp(X), axis=1).reshape(-1,1)

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return (np.exp(input)/np.sum(np.exp(input), axis=1).reshape(-1,1)) * (1 - np.exp(input)/np.sum(np.exp(input), axis=1).reshape(-1,1)) * delta

    def update_parameters(self, gradient_step=0.001):
        pass

class LogSoftmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.log(np.exp(X)/np.sum(np.exp(X), axis=1).reshape(-1,1))

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return (1 - np.exp(input)/np.sum(np.exp(input), axis=1).reshape(-1,1)) * delta

    def update_parameters(self, gradient_step=0.001):
        pass

class ReLu(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        return np.where(X>0, X, 0)

    def backward_delta(self, input, delta):
        return np.where(input>0, 1, 0) * delta

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step=0.001):
        pass
