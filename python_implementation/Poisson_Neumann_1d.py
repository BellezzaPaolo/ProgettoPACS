"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
dde.backend.set_default_backend('pytorch')
dde.config.set_random_seed(123)

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return dy_xx - 2


def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], -1)


def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)


def func(x):
    return (x + 1) ** 2


geom = dde.geometry.Interval(-1, 1)
bc_l = dde.icbc.DirichletBC(geom, func, boundary_l)
bc_r = dde.icbc.NeumannBC(geom, lambda X: 2 * (X + 1), boundary_r)
data = dde.data.PDE(geom, pde, [bc_l, bc_r], 16, 2, solution=func, num_test=100)

layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"


dde.config.set_random_seed(123)
netP = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, netP)

budget = int(1e5)

model.compile("paraflow", lr=1e-3, metrics=["l2 relative error"],n_fine = 100)
losshistory, train_state = model.train(iterations = budget, display_every = 5, callbacks = [dde.callbacks.BudgetCallback(budget)])
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)
print(f'iter: {budget}, coarse {model.opt.counter["call_coarse"]},fine:{model.opt.counter["call_fine"]}, mean correction steps: {model.opt.counter["correction_steps"]/model.opt.counter["iterations"]:.2f} and total iterations: {model.opt.counter["iterations"]}')

# dde.config.set_random_seed(123)
# net = dde.nn.FNN(layer_size, activation, initializer)
# model = dde.Model(data, net)
# model.compile("sgd", lr=1e-3, metrics=["l2 relative error"]) # less than lr = 2e-2
# losshistory, train_state = model.train(iterations = budget, batch_size= 10, display_every=int(budget//100), callbacks = [dde.callbacks.BudgetCallback(budget)])
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# learning_rate = [1e-2, 1e-3, 1e-4]
# n_fine_vec = [10, 50, 100, 500, 1000]
# iterate = 1e5
