import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_data(data,labels=None):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    if labels is not None:
        labels = labels.reshape(-1)
    cols,marks = ["red", "green", "blue", "orange", "black", "cyan"],[".","+","*","o","x","^"]
    if labels is None:
        plt.scatter(data[:,0],data[:,1],marker="x")
        return
    for i,l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(data[labels==l,0],data[labels==l,1],c=cols[i],marker=marks[i])

def plot_frontiere(data,f,seq,step=20):
    """ Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid,x,y=make_grid(data=data,step=step)

    plt.contourf(x,y,f(grid,seq).reshape(x.shape),colors=('gray','blue'),levels=[-1,0,1])

def make_grid(data=None,xmin=-5,xmax=5,ymin=-5,ymax=5,step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:,0]),  np.min(data[:,0]), np.max(data[:,1]), np.min(data[:,1])
    x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02):
    """ Generateur de donnees,
        :param centerx: centre des gaussiennes
        :param centery:
        :param sigma: des gaussiennes
        :param nbex: nombre d'exemples
        :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
        :param epsilon: bruit dans les donnees
        :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type==0:
         #melange de 2 gaussiennes
         xpos=np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//2)
         xneg=np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//2)
         data=np.vstack((xpos,xneg))
         y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))
    if data_type==1:
        #melange de 4 gaussiennes
        xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))

    if data_type==2:
        #echiquier
        data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
        y=np.ceil(data[:,0])+np.ceil(data[:,1])
        y=2*(y % 2)-1
    # un peu de bruit
    data[:,0]+=np.random.normal(0,epsilon,nbex)
    data[:,1]+=np.random.normal(0,epsilon,nbex)
    # on mélange les données
    idx = np.random.permutation((range(y.size)))
    data=data[idx,:]
    y=y[idx]
    return data,y.reshape(-1, 1)

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray",aspect='auto')

def show_usps_compare(x, x2):
    plt.figure()
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.imshow(x.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.title(f'Real image')
    plt.subplot(1, 2, 2) # row 1, col 2 index 1
    plt.imshow(x2.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.title(f'AutoEncodeur image')
    plt.show()

def predict_regression(X, seq):
    return seq.forward(X)

def predict_linear(X, seq):
    res = seq.forward(X)
    return np.where(res > 0.5, 1, 0)

def accuracy_linear(seq, X, Y):
    Yhat = predict_linear(X, seq)
    return np.where(Yhat == Y, 1, 0).sum()/len(Y), Yhat

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size,10))
    one_hot_Y[np.arange(Y.size),Y]=1

    return one_hot_Y

def accuracy_argmax(x, y, net):
    yhat = net.forward(x)
    preds = np.argmax(yhat, axis=1)
    y = np.argmax(y, axis=1)
    
    return np.where(preds == y, 1, 0).sum()/len(y)
 
def show_mnist(x):
    img = x.reshape(28,28)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

def show_mnist_compare(x, x2):
    img = x.reshape(28,28)
    img2 = x2.reshape(28,28)
    plt.figure()
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.imshow(img, cmap='gray')
    plt.title(f'Real image')
    plt.subplot(1, 2, 2) # row 1, col 2 index 1
    plt.imshow(img2, cmap='gray')
    plt.title(f'AutoEncodeur image')
    plt.show()


def show_mnist_compare_multi(x, x2):
    plt.figure(figsize=(21,7))

    len_img = len(x)

    for i in range(len_img):
        img = x[i].reshape(28,28)
        img2 = x2[i].reshape(28,28)
        plt.subplot(2, len_img, i+1) # row 1, col 2 index 1
        plt.axis("off")
        plt.imshow(img, cmap='gray', aspect='auto', interpolation='nearest')
        plt.subplot(2, len_img, i+len_img+1) # row 1, col 2 index 1
        plt.axis("off")
        plt.imshow(img2, cmap='gray', aspect='auto', interpolation='nearest')
    plt.show()

def add_noise_gaussien(data, mean=0, std=1, p=0.1):
    gaus_noise = np.random.normal(mean, std, data.shape)
    noise_img = data + p * gaus_noise
    return noise_img 

def add_noise_pepper(data, p=0.1):
    out = data + np.random.choice([0, 1], size=data.shape, p=[1-p, p])
    return np.where(out > 1,1,out)

def show_prediction(seq, X):
    plt.figure(figsize=(21,7))

    for ind_x in range(len(X)):
        plt.subplot(2, int(len(X)/2), ind_x+1) 
        show_usps(X[ind_x])
        plt.title(f'Prediction : {np.argmax(seq.forward(np.array([X[ind_x]])))}')
    plt.show()