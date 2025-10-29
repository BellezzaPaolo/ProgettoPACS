"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
from deepxde import globals
import numpy as np
dde.backend.set_default_backend('pytorch')
dde.config.set_random_seed(123) 

# Define sine function
sin = dde.backend.sin

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)

    return - dy_xx - dy_yy - 8* np.pi**2 *sin(2* np.pi * x[:, 0:1]) * sin(2*np.pi * x[:, 1:2])


def boundary(_, on_boundary):
    return on_boundary


def func(x):
    return np.sin(2*np.pi * x[:, 0:1]) * np.sin(2*np.pi * x[:, 1:2])

globals.iterazione = 0
geom = dde.geometry.Rectangle([0, 0], [1, 1])
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)
data = dde.data.PDE(geom, pde, bc, 20**2, 20*4, solution=func, num_test=60**2)

layer_size = [2] + [150] * 3 + [1]
activation = "sin"
initializer = "Glorot uniform"

budget = 1000

dde.config.set_random_seed(123)
netP = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, netP)
model.compile("sgd", lr=1e-3, metrics=["l2 relative error"])#,n_fine = 100)
losshistory, train_state = model.train(iterations= budget, display_every= 5000//10, callbacks = [dde.callbacks.BudgetCallback(budget)])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
print(f'iter: {globals.iterazione}, coarse {globals.coarse},fine:{globals.fine}')
