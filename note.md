Now the first 3 definitions of functions:
```bash
def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], -1)


def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)


def func(x):
    return (x + 1) ** 2
```
are in the pde.py file that if it's in the build folder don't need to be added in the path folder, else yes (dove mettiamo, genera __pychace__).

The other part is in the main:
```bash
geom = dde.geometry.Interval(-1, 1)
bc_l = dde.icbc.DirichletBC(geom, func, boundary_l)
```

but the lambda function and the list need to be handled in C++:
```bash
bc_r = dde.icbc.NeumannBC(geom, lambda X: 2 * (X + 1), boundary_r)
data = dde.data.PDE(geom, pde, [bc_l, bc_r], 16, 2, solution=func, num_test=100)
```
Move everything to the pde.py file in order to modify only one place when changing the example?

Then the final object needs to be converted in C++:
```bash
py::object obj = ...;
MyClass *cls = obj.cast<MyClass *>();
```
So remake the model class of deepxde:
- has 2 attributes (data and NN) both classes of deepxde
- 2 main method (compile and train)

How can I implement paraflow vs GD?
- like pytorch: make a class optimizer and 2 sons (paraflow and GD)
- simply 2 methods of the class model: train_paraflow and train_GD