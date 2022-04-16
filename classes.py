import numpy as np
from init_module import Module

class Linear(Module):
    def __init__(self, input, output, biais=True):
        super().__init__()
        self.input = input
        self.output = output
        self._parameters = np.random.random((input, output))
        self._biais = np.random.random((1, output))
        self._gradient_b = np.zeros((1, output))
        self._gradient = np.zeros((input, output))
    
    def forward(self, X):
        return np.matmul(X,self._parameters) + self._biais

    def zero_grad(self):
        self._gradient = np.zeros((self.input,self.output))
        self._gradient_b = np.zeros((1,self.output))

    def backward_update_gradient(self, input, delta):
        delta = delta.reshape((delta.shape[0],-1))
        self._gradient = self._gradient + np.dot(input.T,delta)
        self._gradient_b = delta.sum(axis=0)

    def backward_delta(self, input, delta):
        return np.matmul(delta,self._parameters.T)

    def update_parameters(self, gradient_step=0.001):
        self._parameters -= gradient_step * self._gradient
        self._biais -= gradient_step * self._gradient_b

class Sequential():
    def __init__(self, layers=None) -> None:
        self.layers = layers

    def add_end_layer(self, layer):
        self.layers.append(layer)

    def add_start_layer(self, layer):
        self.layers.insert(0, layer)

    def forward(self, X):
        self.inputs_vectors = []
        self.inputs_vectors.append(X)
        for ind in range(0,len(self.layers)):
            self.inputs_vectors.append(self.layers[ind].forward(self.inputs_vectors[-1]))
        
        return self.inputs_vectors[-1]

    def backward(self, bsme):
        reverse_layers = self.layers[::-1]
        reverse_vectors = self.inputs_vectors[::-1]
        delta_val = bsme 
        
        #Others backwards
        for ind in range(0,len(reverse_layers)):
            reverse_layers[ind].backward_update_gradient(reverse_vectors[ind+1], delta_val)
            delta_val = reverse_layers[ind].backward_delta(reverse_vectors[ind+1], delta_val)

    def update_parameters(self, gradient_step=0.001):
        for l in self.layers:
            l.update_parameters(gradient_step=gradient_step)
            l.zero_grad()


class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride):
        super().__init__()
        self.k_size = k_size
        self.chan_in = chan_in 
        self.chan_out = chan_out
        self.stride = stride
        self._parameters = np.random.random((k_size, chan_in, chan_out))
        self._gradient = np.zeros((k_size, chan_in, chan_out))

    def forward(self, batch, length, chan_in):
        res = np.zeros(batch, (length-self.k_size)/self.stride +1,self.chan_out)

        
    def backward_delta(self, input, delta):
        return super().backward_delta(input, delta)

    def backward_update_gradient(self, input, delta):
        return super().backward_update_gradient(input, delta)

    def update_parameters(self, gradient_step=0.001):
        return super().update_parameters(gradient_step)

class MaxPool1D(Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        pass

    def backward_delta(self, input, delta):
        return super().backward_delta(input, delta)

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step=0.001):
        pass

class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        batch = X.shape[0]
        return np.reshape((batch,-1))

    def backward_delta(self, input, delta):
        return input * delta

    def backward_update_gradient(self, input, delta):
        pass

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