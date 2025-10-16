# ProgettoPACS
This subfolder contains 4 files that enable the pyhton implementation of the ParaFlow method into the library [deepXDE](https://github.com/lululxvi/deepxde).
To make it work:
- Clone the [deepXDE repository](https://github.com/lululxvi/deepxde)
- move this 4 files in their locations with this commands and in case replace the existing file:
  ```bash
  mv Poisson_Neumann_1d.py deepxde
  mv paraflow.py deepxde/deepxde/optimizers/pytorch/
  mv optimizer.py deepxde/deepxde/optimizers/pytorch/
  mv model.py deepxde/deepxde/
  ```
In particular:
- Poisson_Neumann_1d.py contains a working example using the paraflow algorithm
- paraflow.py contains the class that actually implements the method
- model.py and optimizer.py are small modifications of the 2 existing files in the deepXDE source only to link the paraflow class with the rest of the library
  
To make it work the requiements of deepXDE are already in the [requirements.txt](https://github.com/lululxvi/deepxde/blob/master/requirements.txt) and the backend for which has been implemented paraflow, is pytorch. So needs to be selected that backend.
