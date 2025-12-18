#include <iostream>
#include <pybind11/embed.h>
#include <functional>
#include "FNN.hpp"
#include "optimizer/Gradient_Descent.hpp"
#include "optimizer/ParaflowS.hpp"
//#include "Model.hpp"

namespace py = pybind11;

int main(){
    // py::scoped_interpreter guard{}; // start the interpreter and keep it alive
    // py::module_ sys = py::module_::import("sys");
    // // py::print(sys.attr("path"));
    // sys.attr("path").attr("append")("/home/paolo/Desktop/ProgettoPACS"); // add to the working directory to reach problem_settings.py
    // // py::print(sys.attr("path"));
    // py::module_ dde = py::module_::import("deepxde"); //import deepxde
    // py::module_ problem_settings = py::module_::import("problem_settings"); // import problem informations like domain, pde, bc,...

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

    // initilize the FNN class
    std::srand(42);//(unsigned int)) time(0));

    constexpr Initializer_bias Ib = Initializer_bias::One;
    constexpr Initializer_weight Iw = Initializer_weight::He_Norm;
    constexpr activation_type A = activation_type::tanh;

    std::vector<int> layer_size = {1, 4, 4, 1};

    FNN<A> net(layer_size);

    net.initialize<Iw, Ib>();

    net.print();

    vector input = vector::Random(1);

    vector a = net.forward(input);
    std::cout << std::endl;
    std::cout << "Input: " << input << std::endl;
    std::cout << "Output: " << a << std::endl;

    vector b = vector::Ones(3);
    vector c = vector::Ones(3);

    std::cout << b * c << std::endl;

    return 0;
}