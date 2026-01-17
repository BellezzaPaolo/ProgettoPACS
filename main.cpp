#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <functional>
#include <memory>
#include "FNN.hpp"
#include "boundary_condition/Dirichlet_BC.hpp"
#include "optimizer/Gradient_Descent.hpp"
#include "optimizer/ParaflowS.hpp"
//#include "Model.hpp"

namespace py = pybind11;

int main(){
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive
    py::module_ sys = py::module_::import("sys");
    // Add conda environment paths
    // sys.attr("path").attr("append")("/home/paolo/miniforge3/envs/pacs/lib/python3.11/site-packages");
    // sys.attr("path").attr("append")("/home/paolo/miniforge3/envs/pacs/lib/python3.11");
    // sys.attr("path").attr("append")("/home/paolo/Desktop/ProgettoPACS");
    //import deepxde
    py::module_ dde = py::module_::import("deepxde"); 
    // py::module_ problem_settings = py::module_::import("problem_settings"); // import problem informations like domain, pde, bc,...

    // geometry
    std::vector<std::array<double, 2>> vertices = {{0.0, 0.0}, {1.0, 0.0}, {1.0, -1.0}, {-1.0, -1.0}, {-1.0, 1.0}, {0.0, 1.0}};

    py::object geom = dde.attr("geometry").attr("Polygon")(vertices);

    // boundary conditions
    std::function<bool(const vector&, bool)> on_boundary = [](const vector&, bool on_bc){return on_bc;};
    std::function<matrix(const matrix&)> func = [](const matrix& x){return matrix::Ones(x.rows(), 1);};

    std::vector<std::shared_ptr<Boundary_Condition>> bc_vector;
    bc_vector.push_back(std::make_shared<Dirichlet_BC>(geom, func, on_boundary));

    matrix prova = (matrix(4, 2) << 0.5, 0.0, 0.6, 0.0, 0.0, -1.0, 0.5, -1.0).finished();
    matrix output = (matrix(4, 1) << 2.0, 0.0, 3.0, 0.1).finished();

    matrix error = bc_vector[0]->error(prova, prova, output, 0, 3);

    std::cout << "data points: "<< prova << std::endl;
    std::cout << "ouput NN: " << output << std::endl;
    std::cout << "errors: " << error << std::endl;
    // py::list bc_list;
    // bc_list.append( dde.attr("icbc").attr("Dirichlet_BC")(geom, problem_settings.attr("func_l"), problem_settings.attr("boundary_l")));
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
    // std::srand(42);//(unsigned int)) time(0));

    // constexpr Initializer_bias Ib = Initializer_bias::One;
    // constexpr Initializer_weight Iw = Initializer_weight::He_Norm;
    // constexpr activation_type A = activation_type::tanh;

    // std::vector<int> layer_size = {1, 4, 4, 1};

    // FNN<A> net(layer_size);

    // net.initialize<Iw, Ib>();

    // net.print();

    // vector input = vector::Random(1);

    // vector& a = net.forward(input);
    // std::cout << std::endl;
    // std::cout << "Input: " << input << std::endl;
    // std::cout<<std::endl;
    // std::cout << "Output: " << a << std::endl;


    return 0;
}