import numpy as np
from tqdm import tqdm

class Optim():
    def __init__(self, net, loss, eps=0.001) -> None:
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y, verbose=False):
        yhat = self.net.forward(batch_x)
        loss = self.loss.forward(batch_y, yhat)
        if verbose:
            print("LOSS : ", loss.mean())
        delta_loss = self.loss.backward(batch_y, yhat)
        self.net.backward(delta_loss)
        self.net.update_parameters(gradient_step=self.eps)

def SGD(net, loss, x, y, batch_size, max_iter=1000, eps=0.001, verbose=False):
        optim = Optim(net, loss, eps)
        # Liste de variables pour simplifier la création des batchs
        card = x.shape[0]
        nb_batchs = card//batch_size
        inds = np.arange(card)

        # Création des batchs
        np.random.shuffle(inds)
        batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

        for i in (range(max_iter)):
            # On mélange de nouveau lorsqu'on a parcouru tous les batchs
            if i%nb_batchs == 0:
                np.random.shuffle(inds)
                batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

            # Mise-à-jour sur un batch
            batch = batchs[i%(nb_batchs)]
            optim.step(x[batch], y[batch], verbose=verbose)
