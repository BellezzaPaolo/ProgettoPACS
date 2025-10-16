#python file to define everything concerning the settings of the problem
import deepxde as dde

# domain is the interval (a,b)
a = -1
b = 1

# that solves this pde
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return dy_xx - 2

# that has on the left boundary
def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], -1)

# this Dirichlet boundary condition
def func_l(x):
    return (x + 1) ** 2

# and on the right boundary
def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)

# this Neumann condition
def func_r(x):
    return 2 * (x + 1)

# the exact solution
def func_ex(x):
    return (x + 1) ** 2

# shape of the neural network
layer_size = [1, 50, 50, 50, 1]

# activation function of the NN
activation = "tanh"

# initializator method for the NN
initializer = "Glorot uniform"