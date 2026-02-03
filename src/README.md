
# C++ / LibTorch implementation (pybind11 + DeepXDE)

This folder contains the C++ implementation of the project.

The key idea is:
- use **LibTorch** for the neural network, autograd, and optimizers
- use **pybind11 (embedded Python)** to call **DeepXDE geometry** utilities (sampling points, `geom.on_boundary`, etc.)
- keep the training pipeline close to the DeepXDE API style (`Pde` + `Model` + optimizers returning a `Result`)

This project currently builds two executables:
- [src/test.cpp](test.cpp): experiment sweep (SGD + ParaFlowS)
- [src/PoissonLshape.cpp](PoissonLshape.cpp): single run / demo

## Contents

- [src/test.cpp](test.cpp): experiment sweep (SGD + ParaFlowS)
- [src/PoissonLshape.cpp](PoissonLshape.cpp): single run / demo
- [src/Pde.hpp](Pde.hpp), [src/Pde.cpp](Pde.cpp): dataset/geometry wrapper + batching + loss assembly
- [src/Model.hpp](Model.hpp): DeepXDE-like façade (`compile`, `train`, `save_data`, `plot_loss_history`)
- [src/FNN.hpp](FNN.hpp): fully-connected network + initializers
- [src/operator/Differential_Operators.hpp](operator/Differential_Operators.hpp): autograd-based operators (e.g. Laplacian)
- [src/optimizer](optimizer): `Optimizer` interface + `Gradient_Descent` (SGD) + `ParaflowS`
- [src/boundary_condition](boundary_condition): boundary-condition base + Dirichlet BC

Outputs are written to [src/results](results) (CSV files).

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

There is also a top-level [requirements.txt](../requirements.txt) for the Python dependencies used in the project.

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
cd ..
doxygen Doxyfile
```

Generated outputs are written under `html/` and `latex/` at the repository root.

