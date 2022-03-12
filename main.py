import classes as cl
import numpy as np

x = np.random.random((50, 13))
y = np.random.random((50, 3))
yhat = np.random.random((50, 3))

mse = cl.MSELoss()

linear = cl.Linear(3,2)

linear
