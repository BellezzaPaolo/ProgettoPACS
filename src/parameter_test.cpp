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
#include "pde/Pde.hpp"
#include "Model.hpp"

namespace py = pybind11;

int main(){
    /** @brief Start the embedded Python interpreter and keep it alive. */
    py::scoped_interpreter guard{};
    
    py::module_ os = py::module_::import("os");
    os.attr("environ")["DDE_BACKEND"] = "pytorch";

    /** @brief Import DeepXDE after backend is set. */
    py::module_ dde = py::module_::import("deepxde");

    /** @brief Set random seed for reproducibility (Python side). */
    dde.attr("config").attr("set_random_seed")(123); 

    /** @brief Geometry definition (L-shaped polygon). */
    std::vector<std::array<double, 2>> vertices = {{0.0, 0.0}, {1.0, 0.0}, {1.0, -1.0}, {-1.0, -1.0}, {-1.0, 1.0}, {0.0, 1.0}};

    py::object geom = dde.attr("geometry").attr("Polygon")(vertices);

    /** @brief Boundary conditions. */
    std::function<bool(const tensor&, bool)> on_boundary = [](const tensor&, bool on_bc){ return on_bc; };

    /**
     * @brief Dirichlet target: zero function.
     * @details Returns a `(N,1)` tensor of zeros.
     */
    std::function<tensor(const tensor&)> func = [](const tensor& x){
        return torch::zeros({x.size(0), 1}, x.options());
    };

    std::vector<std::shared_ptr<Boundary_Condition>> bc_vector;
    bc_vector.push_back(std::make_shared<Dirichlet_BC>(geom, func, on_boundary));

    /** @brief Neural network (needed to build an autograd graph for the PDE residual). */
    auto net = std::make_shared<FNN<Activation::Tanh>>(std::vector<int64_t>{2, 50, 50, 50, 50, 1});
    net->to(torch::kFloat32);
    net->initialize<Initializer_weight::Glorot_Uniform, Initializer_bias::Constant>(0.0);

    std::function<std::vector<tensor>(const tensor&, const tensor&)> pde_equation =
        [](const tensor& inputs, const tensor& outputs) {
            tensor residual = - differential_operators::laplacian(outputs, inputs, /*component=*/0) -1;
            return std::vector<tensor>{residual};
        };

    auto pde = std::make_shared<Pde>(geom, pde_equation, bc_vector, /*Num_domain=*/1200, /*Num_boundary=*/120, /*Num_test=*/1500);

    const std::vector<int> n_fine = {10, 50, 100, 500, 1000, 2000};
    const std::vector<double> lr = {0.1, 0.01, 0.001, 0.0001};
    const std::vector<int> budgets = {int(1e5),int(1e6),int(1e7),int(1e8)};
    const int batch_size = 1448/2;

    /**
     * @brief Write results in the same CSV schema used by the Python experiments.
     * @details Path is anchored to the source tree even when running from `build/`.
     */
    const auto project_root = std::filesystem::path(__FILE__).parent_path();
    std::string filename = "Poisson_Lshape_" + std::to_string(batch_size) + ".csv";
    const auto csv_path = project_root /"results" / filename;

    Model<FNN<Activation::Tanh>> model(pde, net);

    for(size_t i=0; i< lr.size();++i){
        for(size_t j = 0; j < budgets.size();++j){
            dde.attr("config").attr("set_random_seed")(123); 
            net->initialize<Initializer_weight::Glorot_Uniform, Initializer_bias::Constant>(0.0);
            model.compile("SGD", lr[i], /*loss=*/"MSE", /*verbose=*/false);
            Result res = model.train(budgets[j], batch_size, /*budget=*/budgets[j], false);
            model.save_data(csv_path, res);

            for(size_t k = 0; k < n_fine.size(); ++k){
                dde.attr("config").attr("set_random_seed")(123); 
                net->initialize<Initializer_weight::Glorot_Uniform, Initializer_bias::Constant>(0.0);
                model.compile("ParaFlowS", lr[i],/*n_fine = */ n_fine[k], /*loss=*/"MSE", /*verbose=*/false);
                Result res = model.train(budgets[j], batch_size, /*budget=*/budgets[j], false);
                model.save_data(csv_path, res);

            }
            std::cout << "Ended test for learning rate = " << lr[i] << " and budget = " << budgets[j] << std::endl;

        }
    }

    return 0;
}