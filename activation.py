from init_module import Module


class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        #delta = delta.reshape((delta.shape[0],-1))
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