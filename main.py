from turtle import backward
import classes as cl
import numpy as np

x = np.random.random((500, 20))
y = np.random.random((500, 2))

mse = cl.MSELoss()

linear = cl.Linear(20,2)

input()
max_iter = 0#4000

for iter in range(max_iter):
    #Calcul forward
    yhat = linear.forward(x)
    loss = mse.forward(y,yhat)
    print("Linear Iteration ",iter," : => LOSS : ", loss.sum())
    #Calcul backwards and retro-propagation
    bsme = mse.backward(y,yhat)
    linear.backward_update_gradient(x, bsme)

    #Update paramaters and grad 0 
    linear.update_parameters(gradient_step=10e-5)
    linear.zero_grad()

l1 = cl.Linear(20,10)
t1 = cl.TanH()
l2 = cl.Linear(10,2)
t2 = cl.TanH()
s = cl.Sigmoide()

max_iter2 = 0#100000

for iter in range(max_iter2):
    #Calcul forward
    a1 = l1.forward(x)
    assert a1.shape == (500,10)
    z1 = t1.forward(a1)
    assert z1.shape == (500,10)
    a2 = l2.forward(z1)
    assert a2.shape == (500,2)
    z2 = t2.forward(a2)
    assert z2.shape == (500,2)
    yhat = s.forward(z2)
    assert yhat.shape == (500,2)
    loss = mse.forward(y,yhat)
    print("NN Iteration ",iter," : => LOSS : ", loss.sum())
    #Calcul backwards and retro-propagation
    bmse = mse.backward(y,yhat)
    bs = s.backward_delta(z2,bmse)
    bt2 = t2.backward_delta(a2,bs)
    l2.backward_update_gradient(z1,bt2)
    bl2 = l2.backward_delta(z1,bt2)
    bt1 = t1.backward_delta(a1,bl2)
    l1.backward_update_gradient(x,bt1)
    #Update paramaters and grad 0
    l1.update_parameters(gradient_step=0.01)
    l1.zero_grad()
    l2.update_parameters(gradient_step=0.01)
    l2.zero_grad()
    


seq = cl.Sequentiel([l1,t1,l2,t2,s])


max_iter2 = 0#1000

for iter in range(max_iter2):
    #Calcul forward
    yhat = seq.forward(x)
    loss = mse.forward(y, yhat) 
    print("NN Iteration ",iter," : => LOSS : ", loss.sum())
    #Calcul backwards and retro-propagation
    bmse = mse.backward(y,yhat)
    seq.backward(bmse)
    #Update paramaters and grad 0
    seq.update_parameters()

max_iter2 = 0#10000

for iter in range(max_iter2):
    #Calcul forward
    op = cl.Optim(seq, mse, 0.001)
    op.step(x, y, verbose=True)


print("################### SGD #####################")
cl.SGD(cl.Optim(seq, mse, 0.001), x, y, batch_size=200)
