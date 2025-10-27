"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
dde.backend.set_default_backend('pytorch')
dde.config.set_random_seed(123) 
from deepxde import globals
from deepxde.utils import list_to_str

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return dy_xx - 2


def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], -1)


def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)


def func(x):
    return (x + 1) ** 2

globals.iterazione = 0
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
model.compile("paraflow", lr=1e-3, metrics=["l2 relative error"],n_fine = 100)
losshistory, train_state = model.train(iterations= 1, display_every= 10//10)
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)
print(f'iter: {globals.iterazione}, coarse {globals.coarse},fine:{globals.fine}')


# dde.config.set_random_seed(123)
# net = dde.nn.FNN(layer_size, activation, initializer)
# model = dde.Model(data, net)
# model.compile("sgd", lr=1e-3, metrics=["l2 relative error"]) # less than lr = 2e-2
# losshistory, train_state = model.train(iterations= int(globals.iterazione),display_every=int(globals.iterazione//10))
# # dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# learning_rate = [1e-2, 1e-3, 1e-4]
# n_fine_vec = [10, 50, 100, 500, 1000]
# iterate = 1e5

# with open("test.md", "a") as f:
#         f.write(f'| Budget = {iterate} | GD |  ')
#         for nfine in n_fine_vec:
#             f.write(f' ParaflowS n_fine = {nfine} |')
#         f.write('\n|:--------------:|----:|')
#         for nfine in n_fine_vec:
#             f.write(f' :-----:|')
#         for LR in learning_rate:
            
#             dde.config.set_random_seed(123)
#             net = dde.nn.FNN(layer_size, activation, initializer)
#             model = dde.Model(data, net)
#             model.compile("sgd", lr=LR, metrics=["l2 relative error"])
#             losshistory, train_state = model.train(iterations = int(iterate),display_every=iterate//10)
#             # dde.saveplot(losshistory, train_state, issave=True, isplot=True)
#             f.write(f'\n| lr = {LR:.3e} | {sum(losshistory.loss_train[-1]):.3e}  - {int(iterate):.3e} train loss: {train_state.best_loss_train:.2e}  test loss: {train_state.best_loss_test:.2e} test metric: {list_to_str(train_state.best_metrics):s} |')

#             for N_fine in n_fine_vec:

#                 dde.config.set_random_seed(123)
#                 netP = dde.nn.FNN(layer_size, activation, initializer)
#                 model = dde.Model(data, netP)
#                 model.compile("paraflow", lr=LR, metrics=["l2 relative error"],n_fine = N_fine)
#                 losshistory, train_state = model.train(iterations = int(iterate/N_fine), display_every= int(iterate/(N_fine*10)))
#                 #dde.saveplot(losshistory, train_state, issave=True, isplot=True)

#                 print(f'iter: {globals.iterazione}, coarse {globals.coarse},fine:{globals.fine}, lr: {LR}, n_fine: {N_fine} {globals.mean_correction_steps/int(iterate/N_fine):.2f}')
#                 f.write(f'{sum(losshistory.loss_train[-1]):.3e} -  c:{globals.coarse:.2e} f:{globals.fine:.2e} m: {globals.mean_correction_steps:.2f} train loss: {train_state.best_loss_train:.2e}  test loss: {train_state.best_loss_test:.2e} test metric: {list_to_str(train_state.best_metrics):s} |')

#                 globals.mean_correction_steps = 0
#                 globals.fine = 0
#                 globals.coarse = 0
#                 globals.iterazione = 0