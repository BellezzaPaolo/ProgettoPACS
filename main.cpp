#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <functional>
#include <memory>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include "FNN.hpp"
#include "boundary_condition/Dirichlet_BC.hpp"
#include "optimizer/Gradient_Descent.hpp"
#include "optimizer/ParaflowS.hpp"
#include "Pde.hpp"
//#include "Model.hpp"

namespace py = pybind11;
using namespace autodiff;

int main(){
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive
    //py::module_ sys = py::module_::import("sys");
    // Add conda environment paths
    //sys.attr("path").attr("append")("/home/paolo/miniforge3/envs/pacs/lib/python3.11/site-packages");
    //sys.attr("path").attr("append")("/home/paolo/Desktop/ProgettoPACS");
    //import deepxde
    //py::module_ dde = py::module_::import("deepxde"); 

    // Set random seed for reproducibility
    //dde.attr("config").attr("set_random_seed")(123);
    
    // py::module_ problem_settings = py::module_::import("problem_settings"); // import problem informations like domain, pde, bc,...

    // geometry
    //std::vector<std::array<double, 2>> vertices = {{0.0, 0.0}, {1.0, 0.0}, {1.0, -1.0}, {-1.0, -1.0}, {-1.0, 1.0}, {0.0, 1.0}};

    //py::object geom = dde.attr("geometry").attr("Polygon")(vertices);

    // boundary conditions
    //std::function<bool(const vector&, bool)> on_boundary = [](const vector&, bool on_bc){return on_bc;};
    //std::function<matrix(const matrix&)> func = [](const matrix& x){return matrix::Zero(x.rows(), 1);};

    //std::vector<std::shared_ptr<Boundary_Condition>> bc_vector;
    //bc_vector.push_back(std::make_shared<Dirichlet_BC>(geom, func, on_boundary));

    //matrix prova = (matrix(4, 2) << 0.5, 0.0, 0.6, 0.0, 0.0, -1.0, 0.5, -1.0).finished();
    //matrix output = (matrix(23, 1) << 3.0, 1.0, 0.5, 2.0, 0.0, 3.0, 0.1, 0.8, 0.8, 1.2, 2.4, 1.9, 6.4, 3.0, 1.0, 0.5, 2.0, 0.0, 3.0, 1.0, 0.5, 2.0, 0.0).finished();

    //std::function<std::vector<matrix>(const matrix&, const matrix&)> pde_equation = [](const matrix&, const matrix&){std::vector<matrix> res; res.push_back(matrix::Zero(1,1)); return res;};
    //Pde pde(geom, pde_equation, bc_vector, 3,10, 1500);

    //std::function<double(const matrix&)> MSE = [](const matrix& a){return a.array().square().mean();};

    // // collect all pde data into a single class
    // py::object data = dde.attr("data").attr("PDE")(geom, problem_settings.attr("pde"), bc_list, 16, 2, py::arg("solution") = problem_settings.attr("func_ex"), py::arg("num_test") = 100);

    // initilize the FNN class
    std::srand(42);//(unsigned int)) time(0));

    constexpr Initializer_bias Ib = Initializer_bias::One;
    constexpr Initializer_weight Iw = Initializer_weight::He_Norm;
    constexpr activation_type A = activation_type::tanh;

    // ---------------------------------------------------------------------
    // Forward-mode: Hessian(f, wrt(...), at(...)) using dual2nd.
    // Note: forward-mode Hessian works on a function f; it cannot take a
    // computed dual2nd value u and "recover" Hessian without re-evaluations.
    // ---------------------------------------------------------------------
    std::cout << "\n=== Laplacian via Forward-Mode Hessian (dual2nd) ===" << std::endl;
    std::vector<int> layer_size = {2, 50, 50, 50, 50, 1};

    FNN<A, dual2nd> net(layer_size);

    net.initialize<Iw, Ib>();

    net.print();

    vector_t<dual2nd> input = vector_t<dual2nd>::Random(2);

    vector_t<dual2nd> a = net.forward(input);
    dual2nd y = a(0);
    std::cout << std::endl;
    std::cout << "Input: " << input << std::endl;
    std::cout<<std::endl;
    std::cout << "Output: " << a << std::endl;

    auto f = [&net](const dual2nd& x0, const dual2nd& x1) {
        vector_t<dual2nd> x(2);
        x(0) = x0;
        x(1) = x1;
        return net.forward_mixed(x)(0);
    };

    dual2nd u_fwd;
    Eigen::VectorXd g_fwd;
    Eigen::MatrixXd H_fwd = hessian(
        f,
        detail::wrt(input(0), input(1)),
        detail::at(input(0), input(1)),
        u_fwd,
        g_fwd
    );

    std::cout << "grad(u): " << g_fwd.transpose() << std::endl;
    std::cout << "Hessian(u):\n" << H_fwd << std::endl;
    std::cout << "Laplacian (trace Hessian): " << H_fwd.trace() << std::endl;

    return 0;
}