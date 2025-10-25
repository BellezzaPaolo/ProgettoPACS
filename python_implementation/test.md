<!-- | Budget = 100000.0 | GD |   ParaflowS n_fine = 10 | ParaflowS n_fine = 50 | ParaflowS n_fine = 100 | ParaflowS n_fine = 500 | ParaflowS n_fine = 1000 |
|:--------------:|----:| :-----:| :-----:| :-----:| :-----:| :-----:|
| lr = 1.000e-03 | 4.584e-04  - 1.000e+05 train loss: 4.58e-04  test loss: 4.19e-04 test metric: [1.20e-03] |7.158e-05 -  c:4.51e+04 f:1.00e+05 train loss: 7.16e-05  test loss: 6.10e-05 test metric: [5.71e-04] |1.266e-04 -  c:5.25e+04 f:2.00e+05 train loss: 1.27e-04  test loss: 1.15e-04 test metric: [7.09e-04] |2.693e-04 -  c:5.62e+04 f:3.00e+05 train loss: 2.69e-04  test loss: 2.50e-04 test metric: [1.05e-03] |3.208e-04 -  c:5.69e+04 f:4.00e+05 train loss: 3.21e-04  test loss: 2.96e-04 test metric: [1.09e-03] |3.234e-04 -  c:5.73e+04 f:5.00e+05 train loss: 3.23e-04  test loss: 2.98e-04 test metric: [1.10e-03] |
| lr = 1.000e-04 | 3.323e-03  - 1.000e+05 train loss: 3.32e-03  test loss: 2.12e-03 test metric: [4.52e-03] |4.376e-04 -  c:1.95e+05 f:6.00e+05 train loss: 4.38e-04  test loss: 4.00e-04 test metric: [1.14e-03] |4.000e-04 -  c:2.05e+05 f:7.00e+05 train loss: 4.00e-04  test loss: 3.09e-04 test metric: [3.74e-04] |4.240e-04 -  c:2.10e+05 f:8.00e+05 train loss: 4.24e-04  test loss: 3.19e-04 test metric: [4.02e-04] |1.005e-03 -  c:2.11e+05 f:9.00e+05 train loss: 1.01e-03  test loss: 6.79e-04 test metric: [8.97e-04] |1.347e-03 -  c:2.11e+05 f:1.00e+06 train loss: 1.35e-03  test loss: 1.08e-03 test metric: [7.61e-04] | -->

The number of iterations is counted wrongly

| Budget = 1e5 | GD |   ParaflowS n_fine = 10 | ParaflowS n_fine = 50 | ParaflowS n_fine = 100 | ParaflowS n_fine = 500 | ParaflowS n_fine = 1000 |
|:--------------:|----:| :-----:| :-----:| :-----:| :-----:| :-----:|
| lr = 1e-03 | 4.584e-04  | 7.158e-05 |1.266e-04  |2.693e-04 |3.208e-04 | 3.234e-04  |
| lr = 1e-04 | 3.323e-03 |4.376e-04 |4.000e-04 | 4.240e-04  | 1.005e-03 | 1.347e-03  |

| Budget = 1e5 | GD |   ParaflowS n_fine = 10 | ParaflowS n_fine = 50 | ParaflowS n_fine = 100 | ParaflowS n_fine = 500 | ParaflowS n_fine = 1000 |
|:--------------:|----:| :-----:| :-----:| :-----:| :-----:| :-----:|
| lr = 1.000e-03 | 1.000e+05 |7.158e-05 -  c:4.51e+04 f:1.00e+05 train loss: 7.16e-05  test loss: 6.10e-05 test metric: [5.71e-04] |1.266e-04 -  c:5.25e+04 f:2.00e+05 train loss: 1.27e-04  test loss: 1.15e-04 test metric: [7.09e-04] |2.693e-04 -  c:5.62e+04 f:3.00e+05 train loss: 2.69e-04  test loss: 2.50e-04 test metric: [1.05e-03] |3.208e-04 -  c:5.69e+04 f:4.00e+05 train loss: 3.21e-04  test loss: 2.96e-04 test metric: [1.09e-03] |3.234e-04 -  c:5.73e+04 f:5.00e+05 train loss: 3.23e-04  test loss: 2.98e-04 test metric: [1.10e-03] |
| lr = 1.000e-04 | 3.323e-03  - 1.000e+05 train loss: 3.32e-03  test loss: 2.12e-03 test metric: [4.52e-03] |4.376e-04 -  c:1.95e+05 f:6.00e+05 train loss: 4.38e-04  test loss: 4.00e-04 test metric: [1.14e-03] |4.000e-04 -  c:2.05e+05 f:7.00e+05 train loss: 4.00e-04  test loss: 3.09e-04 test metric: [3.74e-04] |4.240e-04 -  c:2.10e+05 f:8.00e+05 train loss: 4.24e-04  test loss: 3.19e-04 test metric: [4.02e-04] |1.005e-03 -  c:2.11e+05 f:9.00e+05 train loss: 1.01e-03  test loss: 6.79e-04 test metric: [8.97e-04] |1.347e-03 -  c:2.11e+05 f:1.00e+06 train loss: 1.35e-03  test loss: 1.08e-03 test metric: [7.61e-04] |


<!-- | Budget = 100000.0 | GD |   ParaflowS n_fine = 10 | ParaflowS n_fine = 50 | ParaflowS n_fine = 100 | ParaflowS n_fine = 500 | ParaflowS n_fine = 1000 |
|:--------------:|----:| :-----:| :-----:| :-----:| :-----:| :-----:|
| lr = 1.000e-03 | 4.584e-04  - 1.000e+05 train loss: 4.58e-04  test loss: 4.19e-04 test metric: [1.20e-03] |7.158e-05 -  c:4.51e+04 f:1.00e+05 m: 1.51 train loss: 7.16e-05  test loss: 6.10e-05 test metric: [5.71e-04] |1.266e-04 -  c:7.38e+03 f:1.00e+05 m: 0.69 train loss: 1.27e-04  test loss: 1.15e-04 test metric: [7.09e-04] |2.693e-04 -  c:3.76e+03 f:1.00e+05 m: 0.76 train loss: 2.69e-04  test loss: 2.50e-04 test metric: [1.05e-03] |3.208e-04 -  c:7.07e+02 f:1.00e+05 m: 0.54 train loss: 3.21e-04  test loss: 2.96e-04 test metric: [1.09e-03] |3.234e-04 -  c:3.50e+02 f:1.00e+05 m: 0.50 train loss: 3.23e-04  test loss: 2.98e-04 test metric: [1.10e-03] |
| lr = 1.000e-04 | 3.323e-03  - 1.000e+05 train loss: 3.32e-03  test loss: 2.12e-03 test metric: [4.52e-03] |4.376e-04 -  c:1.37e+05 f:1.00e+05 m: 6.52 train loss: 4.38e-04  test loss: 4.00e-04 test metric: [1.14e-03] |4.000e-04 -  c:1.04e+04 f:1.00e+05 m: 2.09 train loss: 4.00e-04  test loss: 3.09e-04 test metric: [3.74e-04] |4.240e-04 -  c:5.01e+03 f:1.00e+05 m: 2.01 train loss: 4.24e-04  test loss: 3.19e-04 test metric: [4.02e-04] |1.005e-03 -  c:8.20e+02 f:1.00e+05 m: 1.10 train loss: 1.01e-03  test loss: 6.79e-04 test metric: [8.97e-04] |1.347e-03 -  c:4.06e+02 f:1.00e+05 m: 1.06 train loss: 1.35e-03  test loss: 1.08e-03 test metric: [7.61e-04] | -->

Budget 1e5, n_fine=[10,50,100,500,1000] lr=[1e-3,1e-4]

| Budget = 1e5 | GD |   ParaflowS n_fine = 10 | ParaflowS n_fine = 50 | ParaflowS n_fine = 100 | ParaflowS n_fine = 500 | ParaflowS n_fine = 1000 |
|:--------------:|----:| :-----:| :-----:| :-----:| :-----:| :-----:|
| lr = 1e-03 | 4.584e-04 | **7.158e-05** | 1.266e-04 | 2.693e-04 | 3.208e-04 | 3.234e-04 |
| lr = 1e-04 | 3.323e-03 | 4.376e-04 | **4.000e-04** | 4.240e-04 | 1.005e-03 | 1.347e-03 |

| Budget = 1e5 | GD |   ParaflowS n_fine = 10 | ParaflowS n_fine = 50 | ParaflowS n_fine = 100 | ParaflowS n_fine = 500 | ParaflowS n_fine = 1000 |
|:--------------:|----:| :-----:| :-----:| :-----:| :-----:| :-----:|
| lr = 1e-03 | 1.000e+05 | c:4.51e+04 f:1.00e+05 | c:7.38e+03 f:1.00e+05 | c:3.76e+03 f:1.00e+05 | c:7.07e+02 f:1.00e+05 | c:3.50e+02 f:1.00e+05 |
| lr = 1e-04 | 1.000e+05 | c:1.37e+05 f:1.00e+05 | c:1.04e+04 f:1.00e+05 | c:5.01e+03 f:1.00e+05 | c:8.20e+02 f:1.00e+05 | c:4.06e+02 f:1.00e+05 |

<!-- | Budget = 100000.0 | GD |   ParaflowS n_fine = 10 | ParaflowS n_fine = 50 | ParaflowS n_fine = 100 | ParaflowS n_fine = 500 | ParaflowS n_fine = 1000 |
|:--------------:|----:| :-----:| :-----:| :-----:| :-----:| :-----:|
| lr = 1.000e-02 | 1.611e-04  - 1.000e+05 train loss: 1.61e-04  test loss: 1.65e-04 test metric: [2.70e-03] |8.123e-05 -  c:3.39e+04 f:1.00e+05 m: 3933.00 train loss: 4.66e-05  test loss: 3.46e-05 test metric: [3.91e-04] |1.992e-04 -  c:6.05e+03 f:1.00e+05 m: 53.00 train loss: 2.23e-05  test loss: 1.96e-05 test metric: [2.55e-04] |9.990e-05 -  c:3.02e+03 f:1.00e+05 m: 17.00 train loss: 9.99e-05  test loss: 9.49e-05 test metric: [1.80e-03] |2.095e-04 -  c:6.02e+02 f:1.00e+05 m: 2.00 train loss: 4.41e-05  test loss: 3.86e-05 test metric: [8.65e-04] |1.601e-04 -  c:3.00e+02 f:1.00e+05 m: 0.00 train loss: 1.60e-04  test loss: 1.64e-04 test metric: [2.70e-03] |
| lr = 1.000e-03 | 4.584e-04  - 1.000e+05 train loss: 4.58e-04  test loss: 4.19e-04 test metric: [1.20e-03] |7.158e-05 -  c:4.51e+04 f:1.00e+05 m: 15097.00 train loss: 7.16e-05  test loss: 6.10e-05 test metric: [5.71e-04] |1.266e-04 -  c:7.38e+03 f:1.00e+05 m: 1380.00 train loss: 1.27e-04  test loss: 1.15e-04 test metric: [7.09e-04] |2.693e-04 -  c:3.76e+03 f:1.00e+05 m: 764.00 train loss: 2.69e-04  test loss: 2.50e-04 test metric: [1.05e-03] |3.208e-04 -  c:7.07e+02 f:1.00e+05 m: 107.00 train loss: 3.21e-04  test loss: 2.96e-04 test metric: [1.09e-03] |3.234e-04 -  c:3.50e+02 f:1.00e+05 m: 50.00 train loss: 3.23e-04  test loss: 2.98e-04 test metric: [1.10e-03] |
| lr = 1.000e-04 | 3.323e-03  - 1.000e+05 train loss: 3.32e-03  test loss: 2.12e-03 test metric: [4.52e-03] |4.376e-04 -  c:1.37e+05 f:1.00e+05 m: 65234.00 train loss: 4.38e-04  test loss: 4.00e-04 test metric: [1.14e-03] |4.000e-04 -  c:1.04e+04 f:1.00e+05 m: 4176.00 train loss: 4.00e-04  test loss: 3.09e-04 test metric: [3.74e-04] |4.240e-04 -  c:5.01e+03 f:1.00e+05 m: 2007.00 train loss: 4.24e-04  test loss: 3.19e-04 test metric: [4.02e-04] |1.005e-03 -  c:8.20e+02 f:1.00e+05 m: 220.00 train loss: 1.01e-03  test loss: 6.79e-04 test metric: [8.97e-04] |1.347e-03 -  c:4.06e+02 f:1.00e+05 m: 106.00 train loss: 1.35e-03  test loss: 1.08e-03 test metric: [7.61e-04] | -->


Budget 1e5, n_fine=[10,50,100,500,1000] lr=[1e-2,1e-3,1e-4]

| Budget = 1e5 | GD |   ParaflowS n_fine = 10 | ParaflowS n_fine = 50 | ParaflowS n_fine = 100 | ParaflowS n_fine = 500 | ParaflowS n_fine = 1000 |
|:--------------:|----:| :-----:| :-----:| :-----:| :-----:| :-----:|
| lr = 1e-02 | 1.611e-04 | **8.123e-05** | 1.992e-04 | 9.990e-05 | 2.095e-04 | 1.601e-04 |
| lr = 1e-03 | 4.584e-04 | **7.158e-05** | 1.266e-04 | 2.693e-04 | 3.208e-04 | 3.234e-04 |
| lr = 1e-04 | 3.323e-03 | 4.376e-04 | **4.000e-04** | 4.240e-04 | 1.005e-03 | 1.347e-03 |

| Budget = 1e5 | GD |   ParaflowS n_fine = 10 | ParaflowS n_fine = 50 | ParaflowS n_fine = 100 | ParaflowS n_fine = 500 | ParaflowS n_fine = 1000 |
|:--------------:|----:| :-----:| :-----:| :-----:| :-----:| :-----:|
| lr = 1e-02 | 1.000e+05 | c:3.39e+04 f:1.00e+05 | c:6.05e+03 f:1.00e+05 | c:3.02e+03 f:1.00e+05 | c:6.02e+02 f:1.00e+05 | c:3.00e+02 f:1.00e+05 |
| lr = 1e-03 | 1.000e+05 | c:4.51e+04 f:1.00e+05 | c:7.38e+03 f:1.00e+05 | c:3.76e+03 f:1.00e+05 | c:7.07e+02 f:1.00e+05 | c:3.50e+02 f:1.00e+05 |
| lr = 1e-04 | 1.000e+05 | c:1.37e+05 f:1.00e+05 | c:1.04e+04 f:1.00e+05 | c:5.01e+03 f:1.00e+05 | c:8.20e+02 f:1.00e+05 | c:4.06e+02 f:1.00e+05 |