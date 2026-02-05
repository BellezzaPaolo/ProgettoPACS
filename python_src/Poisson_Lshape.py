"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import csv
dde.backend.set_default_backend('pytorch')
dde.config.set_random_seed(123)


def pde(x, y):
    # Most backends
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    # Backend jax
    # dy_xx, _ = dde.grad.hessian(y, x, i=0, j=0)
    # dy_yy, _ = dde.grad.hessian(y, x, i=1, j=1)
    return -dy_xx - dy_yy - 1


def boundary(_, on_boundary):
    return on_boundary


geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=1200, num_boundary=120, num_test=1500)

# single training with paraflowS and budget of 1e6
learning_rate = 0.1
budget = int(1e6)

net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("paraflow", lr=learning_rate, n_fine = 100)
losshistory, train_state,data  = model.train(iterations=budget, display_every=10,callbacks = [dde.callbacks.BudgetCallback(budget)])
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# Test the code on different learning rate and budgets

## Training parameters settings
# n_fine = [10, 50, 100, 500, 1000, 2000]
# learning_r = [1e-1, 1e-2, 1e-3, 1e-4]

# budgets = [int(1e5),int(1e6),int(1e7),int(1e8)]
# batch_size = int(1448//2)

## Create results file and write header
# bc_file = batch_size if not None else 'full'
# filename = "./Poisson_Lshape_results_"+str(bc_file)+".csv"

# with open(filename, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(['optimizer_name', "batch_size", 'lr', 'final_budget', 'budget', 'n_fine', 'final_loss', "epochs", "time_train", 'optimizer_counter'])


# # Run experiments
# # NOTE this loop can last some hours because it trains many neural networks
# for lr in learning_r:
#     for b in budgets:
#         # Train with SGD optimizer
#         dde.config.set_random_seed(123)
#         net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
#         model = dde.Model(data, net)
#         model.compile(optimizer='sgd', lr = lr)
#         losshistory, train_state, result_dict  = model.train(iterations=b, batch_size=batch_size, display_every=int(b//100),callbacks = [dde.callbacks.BudgetCallback(b)])
#         #dde.saveplot(losshistory, train_state, issave=True, isplot=True)
#         model.save_data(filename, result_dict, lr, b)
#         print(f'SGD done for lr: {lr:.2e},  budget: {b:.2e}')


#         for nf in n_fine:
#             # Train with paraflow optimizer
#             dde.config.set_random_seed(123)

#             net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
#             model = dde.Model(data, net)
#             model.compile(optimizer='paraflow', lr = lr, n_fine = nf)
#             losshistory, train_state, result_dict  = model.train(iterations=b, batch_size=batch_size, display_every=int(b//100),callbacks = [dde.callbacks.BudgetCallback(b)])
#             #dde.saveplot(losshistory, train_state, issave=True, isplot=True)
#             model.save_data(filename, result_dict, lr, b, nf)
#             print(f'paraflow done for lr: {lr:.2e}, budget: {b:.2e}, n_fine: {nf}')
