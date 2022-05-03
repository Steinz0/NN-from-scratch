import numpy as np
from tqdm import tqdm

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

class Linear(Module):
    def __init__(self, input, output, biais=True):
        super().__init__()
        self.input = input
        self.output = output
        self._parameters = np.random.random((input, output)) - 0.5
        self._gradient = np.zeros((input, output))
        self.withbiais = biais
        if biais:
            self._biais = np.random.random((1, output)) - 0.5
            self._gradient_b = np.zeros((1, output))
        else:
            self._biais = np.zeros((1, output))
            
    def forward(self, X):
        return np.dot(X,self._parameters) + self._biais

    def zero_grad(self):
        self._gradient = np.zeros((self.input,self.output))
        if self.withbiais:
            self._gradient_b = np.zeros((1,self.output))

    def backward_update_gradient(self, input, delta):
        self._gradient = self._gradient + np.dot(input.T,delta)
        if self.withbiais:
            self._gradient_b = self._gradient_b + np.sum(delta, axis=0)

    def backward_delta(self, input, delta):
        return np.dot(delta,self._parameters.T)

    def update_parameters(self, gradient_step=0.001):
        self._parameters -= gradient_step * self._gradient
        if self.withbiais:
            self._biais -= gradient_step * self._gradient_b

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

    def backward(self, bLoss):
        reverse_layers = self.layers[::-1]
        reverse_vectors = self.inputs_vectors[::-1]
        delta_val = bLoss 
        
        #Others backwards
        for ind in range(0,len(reverse_layers)):
            reverse_layers[ind].backward_update_gradient(reverse_vectors[ind+1], delta_val)
            delta_val = reverse_layers[ind].backward_delta(reverse_vectors[ind+1], delta_val)

    def update_parameters(self, gradient_step=0.001):
        for l in self.layers:
            l.update_parameters(gradient_step=gradient_step)
            l.zero_grad()

class Optim():
    def __init__(self, net, loss, eps=0.001) -> None:
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y):
        yhat = self.net.forward(batch_x)
        loss = self.loss.forward(batch_y, yhat)
        delta_loss = self.loss.backward(batch_y, yhat)
        self.net.backward(delta_loss)
        self.net.update_parameters(gradient_step=self.eps)

        return loss
def SGD(net, loss, x, y, batch_size, max_iter=1000, eps=0.001, verbose=False):
        optim = Optim(net, loss, eps)
        # Liste de variables pour simplifier la création des batchs
        card = x.shape[0]
        nb_batchs = card//batch_size
        inds = np.arange(card)
        all_loss = []
        # Création des batchs
        np.random.shuffle(inds)
        batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

        for i in tqdm(range(max_iter)):
            # On mélange de nouveau lorsqu'on a parcouru tous les batchs
            if i%nb_batchs == 0:
                np.random.shuffle(inds)
                batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

            # Mise-à-jour sur un batch
            batch = batchs[i%(nb_batchs)]
            loss = optim.step(x[batch], y[batch])
            all_loss.append(loss.mean())
            if verbose:
                print(f'SGD Iteration {i} => Loss : {loss.mean()}')
        
        return all_loss

def SGDWithTest(net, loss, x, y, xTest, yTest, batch_size, max_iter=1000, eps=0.001, verbose=False):
        optim = Optim(net, loss, eps)
        # Liste de variables pour simplifier la création des batchs
        card = x.shape[0]
        nb_batchs = card//batch_size
        inds = np.arange(card)
        all_loss_train = []
        all_loss_test = []
        # Création des batchs
        np.random.shuffle(inds)
        batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

        for i in tqdm(range(max_iter)):
            # On mélange de nouveau lorsqu'on a parcouru tous les batchs
            if i%nb_batchs == 0:
                np.random.shuffle(inds)
                batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

            # Mise-à-jour sur un batch
            batch = batchs[i%(nb_batchs)]
            loss_train = optim.step(x[batch], y[batch])
            loss_test = loss.forward(net.forward(xTest), yTest)
            all_loss_train.append(loss_train.mean())
            all_loss_test.append(loss_test.mean())
            if verbose:
                print(f'SGD Iteration {i} => Loss Train : {loss_train.mean()} | Loss Test : {loss_test.mean()}')
        
        return all_loss_train, all_loss_test 

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

class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride):
        super().__init__()
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        self._parameters = np.ones((k_size, chan_in, chan_out))
        self._gradient = np.zeros((k_size, chan_in, chan_out))

    def forward(self, batch, length, chan_in):
        # res = np.zeros(batch, (length-self.k_size)/self.stride + 1, self.chan_out)
        for c in range(self.chan_out):
            for i in range(0, length-self.k_size, self.stride):
                linear = np.dot(batch[:, i:self.k_size], self._parameters[c])
                print(linear)
        
    def backward_delta(self, input, delta):
        return super().backward_delta(input, delta)

    def backward_update_gradient(self, input, delta):
        return super().backward_update_gradient(input, delta)

    def update_parameters(self, gradient_step=0.001):
        return super().update_parameters(gradient_step)

class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, batch, length, chan_in):
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
        return X.reshape((X.shape[0],))

    def backward_delta(self, input, delta):
        return input * delta

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step=0.001):
        pass