# ProgettoPACS
This subfolder contains 5 files that enable the pyhton implementation of the ParaFlow method into the library [deepXDE](https://github.com/lululxvi/deepxde) and 3 files for test it.
To install it:
- Clone the [deepXDE repository](https://github.com/lululxvi/deepxde)
- move the files in their locations with this commands and in case replace the existing file:
  ```bash
  mv Poisson_Neumann_1d.py deepxde/
  mv Higham_test.py deepxde/
  mv Smorzato_tset.py deepxde/
  mv Poisson_Lshape.py deepxde/
  mv paraflow.py deepxde/deepxde/optimizers/pytorch/
  mv optimizers.py deepxde/deepxde/optimizers/pytorch/
  mv model.py deepxde/deepxde/
  mv callbacks.py deepxde/deepxde/
  mv pde.py deepxde/deepxde/data/
  ```

  NOTE: the file install.sh is ready to do this 2 passages automatically, by simply executing:
   ```bash
   ./install.sh
   ```
   To clone it uses the SSH key.

In particular:
- Poisson_Neumann_1d.py contains a working example using the paraflow algorithm
- paraflow.py contains the class that actually implements the method
- model.py and optimizer.py are small modifications of the 2 existing files in the deepXDE source only to link the paraflow class with the rest of the library
  
To make it work the requiements of deepXDE are already in the [requirements.txt](https://github.com/lululxvi/deepxde/blob/master/requirements.txt) and the backend for which has been implemented paraflow, is pytorch. So needs to be selected that backend.