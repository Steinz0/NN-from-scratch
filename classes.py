from turtle import forward
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

############################ ACTIVATION #################################

class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        delta = delta.reshape((delta.shape[0],-1))
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
        return np.maximum(X, 0)

    def backward_delta(self, input, delta):
        return np.where(input>0, 1, 0) * delta

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step=0.001):
        pass


############################ ACTIVATION #################################


############################ OPTIMIZER ##################################

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
            if ind != len(reverse_layers) - 1:
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

def BatchGD(net, loss, x, y, epochs=5, eps=0.001, verbose=False):
        optim = Optim(net, loss, eps)
        all_loss = []
        for e in tqdm(range(epochs)):
            loss = optim.step(x, y)
            all_loss.append(loss.mean())
            if verbose:
                print(f'BatchGD Epoch {e+1} => Loss : {loss.mean()}')
        
        return all_loss

def MiniBatchGD(net, loss, x, y, batch_size, epochs=5, eps=0.001, verbose=False):
        optim = Optim(net, loss, eps)
        # Liste de variables pour simplifier la création des batchs
        card = x.shape[0]
        nb_batchs = card//batch_size
        inds = np.arange(card)
        all_loss = []
        # Création des batchs
        np.random.shuffle(inds)
        batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

        for e in tqdm(range(epochs)):
            if verbose:
                print(f'MiniBatchGD Epoch {e+1} :')
            for i in range(len(batchs)):
                batch = batchs[i]
                loss = optim.step(x[batch], y[batch])
                all_loss.append(loss.mean())
                if verbose:
                    print(f'\t Iteration {i} => Loss : {loss.mean()}')
        
        return all_loss

def StochasticGD(net, loss, x, y, epochs=5, eps=0.001, verbose=False):
        optim = Optim(net, loss, eps)

        all_loss = []
        inds = np.arange(len(x))
        np.random.shuffle(inds)
        for e in tqdm(range(epochs)):
            if verbose:
                print(f'SGD Epoch {e+1} :')
            for i in tqdm(inds):
                # Mise-à-jour sur un batch
                xi = np.reshape(x[i], (1,len(x[i])))
                yi = np.reshape(y[i], (1,len(y[i])))

                loss = optim.step(xi, yi)
                all_loss.append(loss.mean())
                if verbose:
                    print(f'SGD Iteration {i} => Loss : {loss.mean()}')
        
        return all_loss

############################ LAYER ##################################

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

class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride):
        super().__init__()
        self._k_size = k_size
        self._chan_in = chan_in
        self._chan_out = chan_out
        self._stride = stride
        self._parameters = np.ones((chan_out, k_size, chan_in)) * 1e-5
        self._gradient = np.zeros((chan_out, k_size, chan_in))
    
    def forward(self, X):
        batch, length, chan_in = X.shape
        res = np.zeros((batch, (length - self._k_size) // self._stride + 1, self._chan_out))

        for ind_x in range(batch):
            for c in range(self._chan_out):
                ind_res = 0
                for i in range(0, length, self._stride):
                    if (i - 1 + self._stride + self._k_size) > length:
                        break
                    res[ind_x, ind_res, c] = np.sum(self._parameters[c] * X[ind_x,i:i+self._k_size,:])

                    ind_res += 1
        return res

    def backward_delta(self, input, delta):
        batch, length, chan_in = input.shape
        res = np.zeros((batch, length, chan_in))

        for ind_x in range(batch):
            for c in range(self._chan_out):
                ind = 0
                for i in range(0, length, self._stride):
                    if (i + self._k_size) > length:
                        break
                    res[ind_x,i:i+self._k_size,:] += self._parameters[c, :, :] * delta[ind_x,ind,c]

                    ind += 1
        return res

    def backward_update_gradient(self, input, delta):
        batch, length, chan_in = input.shape

        for ind_x in range(batch):
            for c in range(self._chan_out):
                ind = 0
                for i in range(0, length, self._stride):
                    if (i + self._k_size) > length:
                        break
                    self._gradient[c, :, :] += delta[ind_x,ind,c] * input[ind_x,i:i+self._k_size,:]

                    ind += 1

    def update_parameters(self, gradient_step=0.001):
        return super().update_parameters(gradient_step)

class Conv1D_2(Module):
    def __init__(self, k_size, chan_in, chan_out, stride):
        super().__init__()
        self.conv_horizon = Conv1D(k_size, chan_in, chan_out, stride)
        self.conv_vertical = Conv1D(k_size, chan_in, chan_out, stride)

    def forward(self, X):
        xreshape = X.reshape(X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])), X.shape[2])
        Xt = np.zeros((xreshape.shape[0], xreshape.shape[1], xreshape.shape[2]))
        for ind in range(len(xreshape)):
            Xt[ind] = xreshape[ind].T
        Xt = Xt.reshape(len(Xt), -1, 1)

        forward_horizon = self.conv_horizon.forward(X)
        forward_vertical = self.conv_vertical.forward(Xt)

        return np.concatenate((forward_horizon, forward_vertical), axis=1)

    def backward_delta(self, input, delta):
        xreshape = input.reshape(input.shape[0], int(np.sqrt(input.shape[1])), int(np.sqrt(input.shape[1])), input.shape[2])
        Xt = np.zeros((xreshape.shape[0], xreshape.shape[1], xreshape.shape[2]))
        for ind in range(len(xreshape)):
            Xt[ind] = xreshape[ind].T
        Xt = Xt.reshape(len(Xt), -1, 1)

        backward_horizon = self.conv_horizon.backward_delta(input, delta[:,:int(delta.shape[1]/2)])
        backward_vertical = self.conv_vertical.backward_delta(Xt, delta[:,int(delta.shape[1]/2):,:])

        return backward_horizon + backward_vertical

    def backward_update_gradient(self, input, delta):
        xreshape = input.reshape(input.shape[0], int(np.sqrt(input.shape[1])), int(np.sqrt(input.shape[1])), input.shape[2])
        Xt = np.zeros((xreshape.shape[0], xreshape.shape[1], xreshape.shape[2]))
        for ind in range(len(xreshape)):
            Xt[ind] = xreshape[ind].T
        Xt = Xt.reshape(len(Xt), -1, 1)

        self.conv_horizon.backward_update_gradient(input, delta[:,:int(delta.shape[1]/2),:])
        self.conv_vertical.backward_update_gradient(Xt, delta[:,int(delta.shape[1]/2):,:])

    def update_parameters(self, gradient_step=0.001):
        self.conv_horizon.update_parameters(gradient_step)
        self.conv_vertical.update_parameters(gradient_step)


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self._k_size = k_size
        self._stride = stride
        self.idx = []

    def forward(self, X):
        batch, length, chan_in = X.shape
        res = np.zeros((batch, (length - self._k_size) // self._stride + 1, chan_in))

        for ind_x in range(batch):
                for c in range(chan_in):
                    ind_res = 0
                    for i in range(0, length, self._stride):
                        if (i + self._k_size) > length:
                            break
                        res[ind_x, ind_res, c] = np.max(X[ind_x,i:i+self._k_size,c])
                        # indmax = np.argmax(input[ind_x,i:i+self._k_size,c])
                        # self.idx.append(indmax)
                        ind_res += 1

        return res

    def backward_delta(self, input, delta):
        batch, length, chan_in = input.shape
        res = np.zeros((batch, length, chan_in))
        
        for ind_x in range(batch):
            for c in range(chan_in):
                ind = 0
                for i in range(0, length, self._stride):
                    if (i + self._k_size) <= length:
                        indmax = np.argmax(input[ind_x,i:i+self._k_size,c])
                        res[ind_x, i+indmax, c] += delta[ind_x, ind, c]
                    ind += 1
        
        return res
        # return delta * res[np.repeat(range(batch),chan_in),self.idx,list(range(chan_in))*batch]

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step=0.001):
        pass

class AvgPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self._k_size = k_size
        self._stride = stride
        self.idx = []

    def forward(self, X):
        batch, length, chan_in = X.shape
        res = np.zeros((batch, (length - self._k_size) // self._stride + 1, chan_in))

        for ind_x in range(batch):
                for c in range(chan_in):
                    ind_res = 0
                    for i in range(0, length, self._stride):
                        if (i + self._k_size) > length:
                            break
                        res[ind_x, ind_res, c] = np.mean(X[ind_x,i:i+self._k_size,c])
                        # indmax = np.argmax(input[ind_x,i:i+self._k_size,c])
                        # self.idx.append(indmax)
                        ind_res += 1

        return res
    
    def backward_delta(self, input, delta):
        batch, length, chan_in = input.shape
        res = np.zeros((batch, length, chan_in))
        
        for ind_x in range(batch):
            for c in range(chan_in):
                ind = 0
                for i in range(0, length, self._stride):
                    if (i + self._k_size) < length:
                        indmax = np.argmax(input[ind_x,i:i+self._k_size,c])
                        res[ind_x, i+self._k_size, c] += delta[ind_x, ind, c]/self._k_size
                    ind += 1
        
        return res
        # return delta * res[np.repeat(range(batch),chan_in),self.idx,list(range(chan_in))*batch]

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step=0.001):
        pass

class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        batch, length, chan_in = X.shape
        return X.reshape((batch, length * chan_in))

    def backward_delta(self, input, delta):
        batch, length, chan_in = input.shape
        return delta.reshape(batch, length, chan_in)

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step=0.001):
        pass