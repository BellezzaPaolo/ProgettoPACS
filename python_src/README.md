# Python implementation of Paraflow

This subfolder contains 5 files that enable the Python implementation of the ParaFlow method into the [deepXDE](https://github.com/lululxvi/deepxde) library and 3 files for testing it.

# Installation
To install it:
- Clone the [deepXDE repository](https://github.com/lululxvi/deepxde)
- Move the files to their respective locations using these commands (replace existing files if necessary):
  ```bash
  mv Higham_test.py deepxde/
  mv Smorzato_test.py deepxde/
  mv Poisson_Lshape.py deepxde/
  mv paraflow.py deepxde/deepxde/optimizers/pytorch/
  mv optimizers.py deepxde/deepxde/optimizers/pytorch/
  mv model.py deepxde/deepxde/
  mv callbacks.py deepxde/deepxde/
  mv pde.py deepxde/deepxde/data/
  ```

  > **NOTE:** The `install.sh` script automates these steps. Simply run:
  > ```bash
  > ./install.sh
  > ```
  > The script uses SSH keys for cloning.

## Requirements
To make it work, the dependencies for deepXDE are listed in its [requirements.txt](https://github.com/lululxvi/deepxde/blob/master/requirements.txt). The ParaFlow optimizer has been implemented for the PyTorch backend, so you need to select that backend.

I used a Conda environment; the library versions are documented in `environment.yml`. To recreate this environment, use:
```bash
conda env create -f environment.yml
conda activate pacs
```

# File Description
In particular:
- `paraflow.py` contains the class that implements the ParaFlow method
- `model.py` and `optimizers.py` are modifications of the corresponding deepXDE files, linking the ParaFlow class with the rest of the library
- `callbacks.py` adds a budget-based callback to compare the ParaFlow optimizer with existing deepXDE optimizers
- `pde.py` modifies the `train_next_batch()` method in the PDE class to enable batch training
- The other 3 files implement test cases. Please read them carefully before running, as some tests require several hours due to training multiple neural networks.

# Tests
The 3 files to test the code should be executed after the installation of DeepXDE. Even if 2 of them doesn't use DeepXDE, import from that folder the callback and the ParaFlowS classes so if you don't want to install DeepXDE adapt the imports. The 3 files contain:
- `Higham_test.py`: A classification of two areas in a domain needs to be obtained by a 10 points data set. This is done by training a neural network with four layers, denoted [2,3,3,2]: input and output layers of two neurons and two hidden layers of three neurons. The sigmoid is used as activation function. This is taken from the example at this [paper](https://doi.org/10.1137/18M1165748).
- `Smorzato_test.py`: consider a neural network of four layers, [1,16,16,1]: input and output layers of one neuron and two hidden layers of 16 neurons. The activation function is the standard sigmoid. The goal is to approximate the function $f (x) = exp(−x) sin(x)$ for $x \in [0, 10]$ using a training set $\{x_i\}_{i=1}^{30}$ obtained by 30 equispaced points in [0, 10] and the corresponding values $f (x_i ), i = 1, . . . , 30$.
- `Poisson_Lshape.py`: The goal is to train a PINN with the structure [2,50,50,50,50,1] (an input layer of two neurons, four hidden layers of 50
neurons, and one output layer of one neuron) for solving the Poisson problem $−\Delta u = 1 $ in an L-shaped domain $\Omega$, with homogeneous Dirichlet boundary conditions. The training set is formed by 120 and 1320 points on $\partial \Omega$ and in $\Omega$, respectively.