import classes as cl
import numpy as np
from mltools import *


############## RANDOM DATA #####################################

trainx = np.random.random((500,10))
trainy = np.random.random((500,1))

############### USPS DATA ######################################
uspsdatatrain = "./data/USPS_train.txt"
uspsdatatest = "./data/USPS_test.txt"
alltrainx,alltrainy = load_usps(uspsdatatrain)
alltestx,alltesty = load_usps(uspsdatatest)
neg = 5
pos = 6
X_train_usps,Y_train_usps = get_usps([neg,pos],alltrainx,alltrainy)
X_test_usps,Y_test_usps = get_usps([neg,pos],alltestx,alltesty)

############### 4 GAUSIENNES DATA ######################################

artix, artiy = gen_arti(data_type=1)
artiy = np.where(artiy == -1, 0, 1)


################### INITIALISATION NN ##################################

l1 = cl.Linear(len(artix[0]),10)
t1 = cl.TanH()
l2 = cl.Linear(10,1)
t2 = cl.TanH()
l3 = cl.Linear(4,1)
s = cl.Sigmoide()

seq = cl.Sequential([l1,t1,l2,s])

mse = cl.MSELoss()

################### PREDICITON FUNCTION ##################################

def predict(X):
    res = np.where(seq.forward(X) > 0.5, 1, 0)
    return res
#linear = cl.Linear(len(alltrainx[0]),1)

#max_iter = 10000

# for iter in range(max_iter):
#     #Calcul forward
#     yhat = linear.forward(train_X)
#     loss = mse.forward(train_y,yhat)
#     print("Linear Iteration ",iter," : => LOSS : ", loss.sum())
#     #Calcul backwards and retro-propagation
#     bsme = mse.backward(train_y,yhat)
#     linear.backward_update_gradient(train_X, bsme)

#     #Update paramaters and grad 0 
#     linear.update_parameters(gradient_step=10e-10)
#     linear.zero_grad()


#max_iter2 = 100000

# for iter in range(max_iter2):
#     #Calcul forward
#     a1 = l1.forward(x)
#     assert a1.shape == (500,10)
#     z1 = t1.forward(a1)
#     assert z1.shape == (500,10)
#     a2 = l2.forward(z1)
#     assert a2.shape == (500,2)
#     z2 = t2.forward(a2)
#     assert z2.shape == (500,2)
#     yhat = s.forward(z2)
#     assert yhat.shape == (500,2)
#     loss = mse.forward(y,yhat)
#     print("NN Iteration ",iter," : => LOSS : ", loss.sum())
#     #Calcul backwards and retro-propagation
#     bmse = mse.backward(y,yhat)
#     bs = s.backward_delta(z2,bmse)
#     bt2 = t2.backward_delta(a2,bs)
#     l2.backward_update_gradient(z1,bt2)
#     bl2 = l2.backward_delta(z1,bt2)
#     bt1 = t1.backward_delta(a1,bl2)
#     l1.backward_update_gradient(x,bt1)
#     #Update paramaters and grad 0
#     l1.update_parameters(gradient_step=0.01)
#     l1.zero_grad()
#     l2.update_parameters(gradient_step=0.01)
#     l2.zero_grad()

max_iter = 1000

for iter in range(max_iter):
    #Calcul forward
    yhat = seq.forward(artix)
    loss = mse.forward(artiy, yhat) 
    print("NN Iteration ",iter," : => LOSS : ", loss.mean())
    #Calcul backwards and retro-propagation
    bmse = mse.backward(artiy, yhat)
    seq.backward(bmse)
    #Update paramaters and grad 0
    seq.update_parameters()

# max_iter2 = 0#10000

# for iter in range(max_iter2):
#     #Calcul forward
#     op = cl.Optim(seq, mse, 0.001)
#     op.step(x, y, verbose=True)


# print("################### SGD #####################")
# cl.SGD(cl.Optim(seq, mse, 0.001), x, y, batch_size=200, verbose=True)


plt.figure()
plot_frontiere(artix, predict)
plot_data(artix, artiy)
plt.show()