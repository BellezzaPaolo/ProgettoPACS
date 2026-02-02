#include <iostream>
#include <vector>
#include <array>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <functional>
#include <memory>
#include <chrono>
#include <torch/torch.h>

#include "FNN.hpp"
#include "boundary_condition/Dirichlet_BC.hpp"
#include "operator/Differential_Operators.hpp"
#include "Losses.hpp"
#include "Pde.hpp"
#include "Model.hpp"

namespace py = pybind11;

int main(){
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive
    
    py::module_ os = py::module_::import("os");
    os.attr("environ")["DDE_BACKEND"] = "pytorch";

    // Optional: inspect Python path if needed
    // py::module_ sys = py::module_::import("sys");

    // Import deepxde after backend is set
    py::module_ dde = py::module_::import("deepxde");

    // Set random seed for reproducibility
    dde.attr("config").attr("set_random_seed")(123); 

    // geometry
    std::vector<std::array<double, 2>> vertices = {{0.0, 0.0}, {1.0, 0.0}, {1.0, -1.0}, {-1.0, -1.0}, {-1.0, 1.0}, {0.0, 1.0}};

    py::object geom = dde.attr("geometry").attr("Polygon")(vertices);

    // boundary conditions
    std::function<bool(const tensor&, bool)> on_boundary = [](const tensor&, bool on_bc){ return on_bc; };

    // Dirichlet target: zero function, returns (N,1) tensor of zeros
    std::function<tensor(const tensor&)> func = [](const tensor& x){
        return torch::zeros({x.size(0), 1}, x.options());
    };

    std::vector<std::shared_ptr<Boundary_Condition>> bc_vector;
    bc_vector.push_back(std::make_shared<Dirichlet_BC>(geom, func, on_boundary));

    // Network (needed to build an autograd graph for the PDE residual)
    auto net = std::make_shared<FNN<Activation::Tanh>>(std::vector<int64_t>{2, 50, 50, 50, 50, 1});
    net->to(torch::kFloat32);
    net->initialize<Initializer_weight::Glorot_Uniform, Initializer_bias::Constant>(0.0);

    std::function<std::vector<tensor>(const tensor&, const tensor&)> pde_equation =
        [](const tensor& inputs, const tensor& outputs) {
            tensor residual = - differential_operators::laplacian(outputs, inputs, /*component=*/0) -1;
            return std::vector<tensor>{residual};
        };

    auto pde = std::make_shared<Pde>(geom, pde_equation, bc_vector, /*Num_domain=*/1200, /*Num_boundary=*/120, /*Num_test=*/1500);

    const double lr = 0.1;
    const int iterations = int(1e5);
    const int batch_size = 0;

    Model<FNN<Activation::Tanh>> model(pde, net);
    model.compile("SGD", lr, /*loss=*/"MSE", /*verbose=*/true);
    // model.compile("ParaFlowS", lr,/*n_fine = */ 100, /*loss=*/"MSE", /*verbose=*/true);
    Result res = model.train(iterations, batch_size, /*budget=*/iterations, true);

    std::cout << "Training done. epoch=" << res.epoch
              << " final_loss=" << res.final_loss
              << " budget_used=" << res.budget_used
              << " time_ms=" << res.total_time_ms
              << std::endl;

    return 0;
}