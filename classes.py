from tabnanny import verbose
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
        return -2*np.sum(y-yhat,axis=1)

class Linear(Module):
    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output
        self._parameters = np.random.random((input, output))
        self._gradient = np.zeros((input, output))
    
    def forward(self, X):
        return np.matmul(X,self._parameters)

    def zero_grad(self):
        self._gradient = np.zeros((self.input,self.output))

    def backward_update_gradient(self, input, delta):
        delta = delta.reshape((delta.shape[0],-1))
        self._gradient = self._gradient + np.dot(input.T,delta)

    def backward_delta(self, input, delta):
        #delta = delta.reshape((delta.shape[0],-1))
        return np.matmul(delta,self._parameters.T)

    def update_parameters(self, gradient_step=0.001):
        self._parameters -= gradient_step * self._gradient

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

class Sequentiel():
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

class Optim():
    def __init__(self, net, loss, eps=0.001) -> None:
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y, verbose=False):
        yhat = self.net.forward(batch_x)
        loss = self.loss.forward(batch_y, yhat)
        if verbose:
            print("LOSS : ", loss.sum())
        delta_loss = self.loss.backward(batch_y, yhat)
        self.net.backward(delta_loss)
        self.net.update_parameters(gradient_step=self.eps)


# def SGD(net, dataset, batch_size, nb_iter):
#     indices = np.arange(dataset.shape[0])
#     np.random.shuffle(indices)
#     print(dataset.shape[0])
#     for i in range(batch_size-1):
        

def SGD(optim, x, y, batch_size, niter=1000):
        # Liste de variables pour simplifier la création des batchs
        card = x.shape[0]
        nb_batchs = card//batch_size
        inds = np.arange(card)

        # Création des batchs
        np.random.shuffle(inds)
        batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

        for i in range(niter):
            # On mélange de nouveau lorsqu'on a parcouru tous les batchs
            if i%nb_batchs == 0:
                np.random.shuffle(inds)
                batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

            # Mise-à-jour sur un batch
            batch = batchs[i%(nb_batchs)]
            print("###### BATCH ",i," ######")
            optim.step(x[batch], y[batch], verbose=True)