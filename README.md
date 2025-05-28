# ProgettoPACS
Contains 4 files that aneble the impelementation of the Paraflow method in the library [deepXDE](https://github.com/lululxvi/deepxde).
To make it work:
- Clone the [deepXDE repository](https://github.com/lululxvi/deepxde)
- move this 4 files in their locations with this commands and in case replace the existing file:
  ```bash
  mv Poisson_Neumann_1d.py deepxde
  mv paraflow.py deepxde/deepxde/optimizers/pytorch/
  mv optimizer.py deepxde/deepxde/optimizers/pytorch/
  mv model.py deepxde/deepxde/
  ```
To make it work the requiements of deepXDE are already in the [requirements.txt](https://github.com/lululxvi/deepxde/blob/master/requirements.txt) and the backend for which has been implemented paraflow, is pytorch. So needs to be selected that backend.
