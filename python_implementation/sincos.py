"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import csv
dde.backend.set_default_backend('pytorch')
dde.config.set_random_seed(123) 

# Define sine function
sin = dde.backend.sin
cos = dde.backend.cos
exp = dde.backend.exp

k0 = np.pi

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)

    return - dy_xx - dy_yy +  (sin(k0 * x[:, 0:1]) - 2 * k0 * cos(k0 * x[:, 0:1]) - 2 * k0**2 * sin(k0 * x[:,0:1])) * sin(k0 * x[:, 1:2]) * exp(-x[:,0:1])


def boundary(_, on_boundary):
    return on_boundary


def func(x):
    return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2]) * np.exp(- x[:,0:1])

geom = dde.geometry.Rectangle([0, 0], [1, 1])
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)
data = dde.data.PDE(geom, pde, bc, 20**2, 20*4, solution=func, num_test=60**2)

layer_size = [2] + [150] * 3 + [100] + [1]
activation = "sin"
initializer = "Glorot uniform"

# budget = 1000

# dde.config.set_random_seed(123)
# netP = dde.nn.FNN(layer_size, activation, initializer)
# model = dde.Model(data, netP)
# model.compile("sgd", lr=1e-4, metrics=["l2 relative error"])#,n_fine = 100)
# losshistory, train_state, result_dict = model.train(iterations= budget, display_every= 100)#, callbacks = [dde.callbacks.BudgetCallback(budget)])
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# Test the code on different learning rate and budgets

# Training parameters settings
n_fine = [10, 50, 100, 500, 1000, 2000]
learning_r = [1e-5, 1e-6]

budgets = [int(1e4),int(1e5),int(1e6),int(1e7)]
batch_size = int(560//2) # full batch = 560

# Create results file and write header
bc_file = batch_size if not None else 'full'
filename = "results/Poisson_expsinsin_results_"+str(bc_file)+".csv"

with open(filename, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['optimizer_name', "batch_size", 'lr', 'final_budget', 'budget', 'n_fine', 'final_loss', "epochs", "time_train", 'optimizer_counter'])


# Train with paraflow optimizer
dde.config.set_random_seed(123)
lr = 1e-4
b = int(1e7)
nf = 500
netP = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, netP)
model.compile("paraflow", lr=lr, metrics=["l2 relative error"], n_fine = nf)
losshistory, train_state, result_dict  = model.train(iterations=b, batch_size=batch_size, display_every=int(b//100),callbacks = [dde.callbacks.BudgetCallback(b)])
#dde.saveplot(losshistory, train_state, issave=True, isplot=True)
model.save_data(filename, result_dict, lr, b, nf)
print(f'paraflow done for lr: {lr:.2e}, budget: {b:.2e}, n_fine: {nf}')

dde.config.set_random_seed(123)
lr = 1e-4
b = int(1e7)
nf = 1000
netP = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, netP)
model.compile("paraflow", lr=lr, metrics=["l2 relative error"], n_fine = nf)
losshistory, train_state, result_dict  = model.train(iterations=b, batch_size=batch_size, display_every=int(b//100),callbacks = [dde.callbacks.BudgetCallback(b)])
#dde.saveplot(losshistory, train_state, issave=True, isplot=True)
model.save_data(filename, result_dict, lr, b, nf)
print(f'paraflow done for lr: {lr:.2e}, budget: {b:.2e}, n_fine: {nf}')

dde.config.set_random_seed(123)
lr = 1e-4
b = int(1e7)
nf = 2000
netP = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, netP)
model.compile("paraflow", lr=lr, metrics=["l2 relative error"], n_fine = nf)
losshistory, train_state, result_dict  = model.train(iterations=b, batch_size=batch_size, display_every=int(b//100),callbacks = [dde.callbacks.BudgetCallback(b)])
#dde.saveplot(losshistory, train_state, issave=True, isplot=True)
model.save_data(filename, result_dict, lr, b, nf)
print(f'paraflow done for lr: {lr:.2e}, budget: {b:.2e}, n_fine: {nf}')

# Run experiments
for lr in learning_r:
    for b in budgets:
        # Train with SGD optimizer
        dde.config.set_random_seed(123)
        netP = dde.nn.FNN(layer_size, activation, initializer)
        model = dde.Model(data, netP)
        model.compile("sgd", lr= lr, metrics=["l2 relative error"])
        losshistory, train_state, result_dict  = model.train(iterations=b, batch_size=batch_size, display_every=int(b//100),callbacks = [dde.callbacks.BudgetCallback(b)])
        #dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        model.save_data(filename, result_dict, lr, b)
        print(f'SGD done for lr: {lr:.2e},  budget: {b:.2e}')


        for nf in n_fine:
            # Train with paraflow optimizer
            dde.config.set_random_seed(123)

            netP = dde.nn.FNN(layer_size, activation, initializer)
            model = dde.Model(data, netP)
            model.compile("paraflow", lr=lr, metrics=["l2 relative error"], n_fine = nf)
            losshistory, train_state, result_dict  = model.train(iterations=b, batch_size=batch_size, display_every=int(b//100),callbacks = [dde.callbacks.BudgetCallback(b)])
            #dde.saveplot(losshistory, train_state, issave=True, isplot=True)
            model.save_data(filename, result_dict, lr, b, nf)
            print(f'paraflow done for lr: {lr:.2e}, budget: {b:.2e}, n_fine: {nf}')
