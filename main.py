from turtle import backward
import classes as cl
import numpy as np

x = np.random.random((500, 20))
y = np.random.random((500, 2))

mse = cl.MSELoss()

linear = cl.Linear(20,2)

linear.forward(x)

max_iter = 500

for iter in range(max_iter):
    #Calcul forward
    yhat = linear.forward(x)
    loss = mse.forward(y,yhat)
    print("Iteration ",iter," : for ", loss.sum())
    #Calcul backwards and retro-propagation
    bsme = mse.backward(y,yhat)
    linear.backward_update_gradient(x, bsme)

    linear.update_parameters(gradient_step=10e-7)
    linear.zero_grad()