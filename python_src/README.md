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
- `Higham_test.py`: This test implements a classification problem of 10 points in the domain $[0,1]^2$. This is taken from the example at this [paper](https://doi.org/10.1137/18M1165748)
- `Smorzato_test.py`:
- `Poisson_Lshape.py`: