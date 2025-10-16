#include <iostream>
#include <pybind11/embed.h>
#include "FNN.hpp"
#include "Optimizer.hpp"
namespace py = pybind11;

class Model{
    private:
        py::object data;
        FNN net;
        Optimizer opt;

    public:
        Model(py::object Data, FNN nn, Optimizer Opt): data(Data), net(nn), opt(Opt){};

};