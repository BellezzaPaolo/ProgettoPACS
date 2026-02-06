# ProgettoPACS
This repository aims to integrate the ParaFlowS method into the library [deepXDE](https://github.com/lululxvi/deepxde). It will be done in C++ using pybind11 to bind python and C++ code. A simple python implementation is contained into the python_src folder. For both implementation refer to their README.md.

## ParaFlowS

ParaFlowS is a sequential optimization algorithm inspired by the Parareal time-parallel method and tailored for gradient-flow–based minimization problems. Instead of computing the full time trajectory, ParaFlowS focuses directly on reaching the minimizer of the objective function. At each iteration, a fine and a coarse propagator are evaluated starting from the same parameter vector. A sequence of corrected candidates is generated using a Parareal-type update, and the process is stopped as soon as the loss function stops decreasing. This adaptive stopping rule avoids unnecessary computations and allows the effective time horizon to be chosen dynamically. Unlike classical Parareal, ParaFlowS does not require parallel execution and operates fully sequentially. The method iteratively updates the parameters until convergence or a maximum number of iterations is reached, providing an efficient trade-off between accuracy and computational cost.

## C++ implementation

See [src/README.md](src/README.md) for:
- build/run instructions (LibTorch + embedded Python)
- project structure
- CSV outputs and Doxygen generation

## python implementation

See [python_src/README.md](python_src/README.md) for:
- build/run instructions
- project installation
- description of the files