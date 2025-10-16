#include <iostream>
#include <pybind11/embed.h>
#include <functional>
#include "Gradient_Descent.hpp"
//#include "Model.hpp"

namespace py = pybind11;

int main(){
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    py::module_ sys = py::module_::import("sys");
    // py::print(sys.attr("path"));
    sys.attr("path").attr("append")("/home/paolo/Desktop/ProgettoPACS"); // add to the working directory to reach pde.py
    // py::print(sys.attr("path"));
    // py::module_ dde = py::module_::import("deepxde"); //import deepxde
    // py::module_ problem_settings = py::module_::import("problem_settings"); // import problem informations like domain, pde, bc,...
    std::cout << "Python executable: " 
            << py::module_::import("sys").attr("executable").cast<std::string>() 
            << std::endl;

    // // geometry
    // py::object geom = dde.attr("geometry").attr("Interval")(problem_settings.attr("a"), problem_settings.attr("b"));

    // // boundary conditions
    // py::list bc_list;
    // bc_list.append( dde.attr("icbc").attr("DirichletBC")(geom, problem_settings.attr("func_l"), problem_settings.attr("boundary_l")));
    // bc_list.append( dde.attr("icbc").attr("NeumannBC")(geom, problem_settings.attr("func_r"),problem_settings.attr("boundary_r")));

    // // collect all pde data into a single class
    // py::object data = dde.attr("data").attr("PDE")(geom, problem_settings.attr("pde"), bc_list, 16, 2, py::arg("solution") = problem_settings.attr("func_ex"), py::arg("num_test") = 100);

    // // import all neural network informations
    // py::object py_layer_size = problem_settings.attr("layer_size");
    // std::vector<int> layer_size;
    // for (auto item : py_layer_size) {
    //     layer_size.push_back(item.cast<int>());
    // }
    // std::string activation = problem_settings.attr("activation").cast<std::string>();
    // std::string initializer = problem_settings.attr("initializer").cast<std::string>();

    // // initilize the FNN class
    // FNN nn(layer_size, activation, initializer);

    std::function<double(std::vector<double>&)> f = [](std::vector<double>& x){return (x[0]+1)*(x[0]+1)+x[1]*x[1];};
    std::function<std::vector<double>(std::vector<double>&)> df = [](std::vector<double>& x){return std::vector<double>{2*x[0]+2, 2*x[1]};};
    int max_iter = 100;
    double tol = 1e-2;
    int dim = 2;
    double lr = 0.01;
    int batch_size = 10;

    Gradient_Descent gd(f, df, max_iter, tol, dim, lr, batch_size);



    return 0;
}