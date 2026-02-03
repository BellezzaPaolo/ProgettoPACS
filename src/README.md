
# C++ / LibTorch implementation (pybind11 + DeepXDE)

This folder contains the C++ implementation of the project.

The key idea is:
- use **LibTorch** for the neural network, autograd, and optimizers
- use **pybind11 (embedded Python)** to call **DeepXDE geometry** utilities (sampling points, `geom.on_boundary`, etc.)
- keep the training pipeline close to the DeepXDE API style (`Pde` + `Model` + optimizers returning a `Result`) in order to be easily and very intuitive to extend.

This project currently builds two executables:
- [src/PoissonLshape.cpp](PoissonLshape.cpp): example presented in the paper [DeepXDE: A Deep Learning Library for Solving Differential Equations](https://doi.org/10.1137/19M1274067), section 4.1.COnsider 2D Poisson problem over the domain $\Omega = [-1,1]^2 \setminus [0,1]^2$:
$$
\begin{cases}
-\Delta u = 1 & \text{in } \Omega,\\
u = 0 & \text{on } \partial\Omega,
\end{cases}
$$
- [src/test.cpp](test.cpp): experiment to test the velocity of convergence with many settings of GD and ParaFlowS, keeping as test case the above problem.

## Contents

- [src/test.cpp](test.cpp): experiment tester (SGD + ParaFlowS)
- [src/PoissonLshape.cpp](PoissonLshape.cpp): single example
- [src/Pde.hpp](Pde.hpp), [src/Pde.cpp](Pde.cpp): dataset/geometry wrapper and handles also batching and loss assembly
- [src/Model.hpp](Model.hpp): DeepXDE-like facade (`compile`, `train`, `save_data`, `plot_loss_history`)
- [src/FNN.hpp](FNN.hpp): fully-connected network and initializers
- [src/operator/Differential_Operators.hpp](operator/Differential_Operators.hpp): autograd-based operators. By now it supports only Laplacian but can be upgraded.
- [src/optimizer](optimizer): `Optimizer` abstract class,`Gradient_Descent` (SGD) and `ParaflowS`
- [src/boundary_condition](boundary_condition): boundary-condition base and Dirichlet BC. By now support only Dirichlet boundary condition but can be upgraded.

Outputs are written to [src/results](results) (CSV files), folder not commited but the code is made deterministic and so reproducible.

## Requirements

### C++ / build dependencies

- CMake **>= 3.28** (as in [src/CMakeLists.txt](CMakeLists.txt))
- A C++17 compiler
- **LibTorch** (C++ PyTorch)
- **pybind11** (for `pybind11::embed`)

### Python runtime dependencies (embedded)

The executable embeds Python and imports DeepXDE, so you need a working Python environment with:
- `deepxde`
- `numpy`, `scipy`, `matplotlib` (and other deps you use)

There is also the [requirements.txt](/requirements.txt) that contains the requirements of DeepXDE.

For more information watch the [../python_src/README.md](../python_src/README.md) and the [deepXDE](https://github.com/lululxvi/deepxde) guide.

## Configuration notes (paths)

The current [src/CMakeLists.txt](CMakeLists.txt) contains **hard-coded paths** for:
- `pybind11_DIR` (Conda environment)
- `CMAKE_PREFIX_PATH` (LibTorch install)
- `PYTHONHOME` and `LD_LIBRARY_PATH` for the custom run targets

If you are on another machine, update these variables to match your setup.

## Build

From the repository root:

```bash
cd src
cmake -S . -B build
cmake --build build -j
```

This produces the `example` executable under `src/build/`.

This produces two executables under `src/build/`:
- `test`
- `poisson_lshape`

## Run

The CMake project defines run targets that set `PYTHONHOME` and `LD_LIBRARY_PATH` so the embedded Python can find its shared libraries:

```bash
cd src
cmake --build build --target run_test
cmake --build build --target run_poissonlshape
```

If you prefer Makefile-style:

```bash
cd src/build
make run_test
make run_poissonlshape
```

## What the executables do

The current entry points are [src/test.cpp](test.cpp) and [src/PoissonLshape.cpp](PoissonLshape.cpp):

- Initializes Python and sets `DDE_BACKEND=pytorch`
- Imports DeepXDE and constructs a polygon geometry
- Builds a PINN (`FNN<Tanh>`) in float32
- Defines the PDE residual using autograd Laplacian

`test.cpp` runs a sweep over learning rates and budgets for:
	- SGD (called `SGD` / `Gradient_Descent` in the code)
	- ParaFlowS with multiple `n_fine`
- Appends one CSV row per training run via `Model::save_data(...)`

`PoissonLshape.cpp` runs a single training configuration (useful as a quick demo / smoke test).

The CSV is written to: [src/results/Poisson_Lshape_0.csv](results/Poisson_Lshape_0.csv)

## Notes

- **PINN input gradients:** batches are cloned/detached and marked `requires_grad(true)` so differential operators (e.g., Laplacian) work.
- **dtype parity:** the pipeline is intended to run in float32 for comparison with DeepXDE defaults.

## Documentation (Doxygen)

The repo includes a [Doxyfile](../Doxyfile). To generate docs:

```bash
doxygen Doxyfile
```

Generated outputs are written under `html/` and `latex/` at the repository root.

