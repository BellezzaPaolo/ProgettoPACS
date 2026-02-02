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
// #include <iostream>
// #include <vector>
// #include <array>
// #include <pybind11/embed.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <functional>
// #include <memory>
// #include <chrono>
// #include <torch/torch.h>
// // #include <autodiff/forward/dual.hpp>
// // #include <autodiff/forward/dual/eigen.hpp>
// // #include <autodiff/reverse/var.hpp>
// // #include <autodiff/reverse/var/eigen.hpp>
// #include "FNN.hpp"
// #include "boundary_condition/Dirichlet_BC.hpp"
// #include "operator/Differential_Operators.hpp"
// // #include "optimizer/Gradient_Descent.hpp"
// // #include "optimizer/ParaflowS.hpp"
// #include "Pde.hpp"
// // #include "Model.hpp"

// namespace py = pybind11;
// // using namespace autodiff;

// int main(){
//     py::scoped_interpreter guard{}; // start the interpreter and keep it alive
    
//     py::module_ os = py::module_::import("os");
//     os.attr("environ")["DDE_BACKEND"] = "pytorch";

//     // Optional: inspect Python path if needed
//     // py::module_ sys = py::module_::import("sys");

//     // Import deepxde after backend is set
//     py::module_ dde = py::module_::import("deepxde");

//     // Set random seed for reproducibility
//     dde.attr("config").attr("set_random_seed")(123); 
     
//     // // py::module_ problem_settings = py::module_::import("problem_settings"); // import problem informations like domain, pde, bc,...

//     // geometry
//     std::vector<std::array<double, 2>> vertices = {{0.0, 0.0}, {1.0, 0.0}, {1.0, -1.0}, {-1.0, -1.0}, {-1.0, 1.0}, {0.0, 1.0}};

//     py::object geom = dde.attr("geometry").attr("Polygon")(vertices);

//     // boundary conditions (Torch-based)
//     std::function<bool(const tensor&, bool)> on_boundary = [](const tensor&, bool on_bc){ return on_bc; };

//     // Dirichlet target: zero function, returns (N,1) tensor of zeros
//     std::function<tensor(const tensor&)> func = [](const tensor& x){
//         return torch::zeros({x.size(0), 1}, x.options());
//     };

//     std::vector<std::shared_ptr<Boundary_Condition>> bc_vector;
//     bc_vector.push_back(std::make_shared<Dirichlet_BC>(geom, func, on_boundary));

//     // Simple network (needed to build an autograd graph for the PDE residual)
//     FNN<Activation::Tanh> net(std::vector<int64_t>{2, 50, 50, 50, 50, 1});
//     net.to(torch::kDouble);
//     net.initialize<Initializer_weight::Glorot_Uniform, Initializer_bias::Constant>(0.0);

//     std::function<std::vector<tensor>(const tensor&, const tensor&)> pde_equation =
//         [](const tensor& inputs, const tensor& outputs) {
//             tensor residual = - differential_operators::laplacian(outputs, inputs, /*component=*/0) -1;
//             return std::vector<tensor>{residual};
//         };

//     Pde pde(geom, pde_equation, bc_vector, /*Num_domain=*/1200, /*Num_boundary=*/120, /*Num_test=*/1500);

//     // Mean Squared Error (scalar tensor). Use this for PDE residuals and BC errors.
//     const std::function<tensor(const tensor&)> MSE = [](const tensor& r) {
//         return r.square().mean();
//     };

//     // -----------------------------
//     // PyTorch Gradient Descent (SGD)
//     // -----------------------------
//     const double lr = 0.1;
//     const int budget = int(1e5);
//     const int batch_size = 0;
//     const double w_pde = 1.0;
//     const double w_bc = 1.0;

//     torch::optim::SGD optimizer(net.parameters(), torch::optim::SGDOptions(lr));

//     // After dataset/geometry construction, we can release the GIL for the whole
//     // training loop (no Python calls are needed inside the loop).
//     py::gil_scoped_release no_gil;

//     int b_used = 0;
//     auto t0 = std::chrono::high_resolution_clock::now();

//     for (int it = 0; it < budget; ++it) {
//         tensor& batch_x = pde.train_next_batch(batch_size);
//         // const int64_t N = batch_x.size(0);

//         // Leaf input for autograd w.r.t. x (needed for Laplacian)
//         tensor x = batch_x.clone().detach().set_requires_grad(true);

//         // Forward
//         tensor u = net.forward(x); // [N,1]

//         // Losses from Pde (PDE losses first, then BC losses)
//         const auto parts = pde.losses(x, u, MSE);

//         // Split into PDE vs BC parts using known counts
//         tensor pde_loss = torch::zeros({}, u.options());
//         tensor bc_loss = torch::zeros({}, u.options());

//         const int64_t n_pde_terms = static_cast<int64_t>(pde_equation(x, u).size());
//         for (int64_t i = 0; i < static_cast<int64_t>(parts.size()); ++i) {
//             if (i < n_pde_terms) {
//                 pde_loss = pde_loss + parts[static_cast<size_t>(i)];
//             } else {
//                 bc_loss = bc_loss + parts[static_cast<size_t>(i)];
//             }
//         }

//         tensor loss = w_pde * pde_loss + w_bc * bc_loss;

//         optimizer.zero_grad();
//         loss.backward();
//         optimizer.step();
//         b_used += batch_x.size(0);

//         if (it % 10 == 0) {
//             std::cout << "it=" << it
//                       << " loss=" << loss.item<double>()
//                       << " pde=" << pde_loss.item<double>()
//                       << " bc=" << bc_loss.item<double>()
//                       << "\n";
//         }
//         if(b_used >= budget){
//             std::cout << it << std::endl;
//             break;
//         }
//     }

//     auto t1 = std::chrono::high_resolution_clock::now();
//     auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
//     std::cout << "Training time: " << ms << " ms\n";

//     // // collect all pde data into a single class
//     // py::object data = dde.attr("data").attr("PDE")(geom, problem_settings.attr("pde"), bc_list, 16, 2, py::arg("solution") = problem_settings.attr("func_ex"), py::arg("num_test") = 100);

//     // initilize the FNN class
//     // std::srand(42);//(unsigned int)) time(0));

//     // torch::manual_seed(0);

//     // std::vector<int64_t> layer_size = {2,32,32,1};

//     // constexpr Initializer_bias Ib = Initializer_bias::Constant;
//     // constexpr Initializer_weight Iw = Initializer_weight::He_Norm;
//     // constexpr Activation A = Activation::Tanh;

//     // FNN<A> net(layer_size);
//     // net.to(torch::kDouble);

//     // net.initialize<Iw, Ib>();

//     // net.print();

//     // // Optimizer
//     // torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));

//     // const int64_t epochs = 500;
//     // torch::Tensor x, y, target, loss;

//     // auto t_start = std::chrono::high_resolution_clock::now();

//     // for(int64_t epoch = 0; epoch < epochs; ++epoch){
//     //     // Input batch
//     //     x = torch::rand({5, 2}, torch::dtype(torch::kDouble));

//     //     // Dummy target for demonstration (here: all ones)
//     //     target = torch::ones({5, 1}, torch::dtype(torch::kDouble));

//     //     // Forward pass
//     //     y = net.forward(x);

//     //     // Compute loss (Mean Squared Error)
//     //     loss = torch::mse_loss(y, target);

//     //     // Backward pass
//     //     optimizer.zero_grad();
//     //     loss.backward();
//     //     optimizer.step();

//     //     if(epoch % 50 == 0){
//     //         std::cout << "Epoch " << epoch
//     //                   << ", loss = " << loss.item<double>() << '\n';
//     //     }
//     // }

//     // auto t_end = std::chrono::high_resolution_clock::now();
//     // auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

//     // std::cout << "\nFinal batch x:\n" << x << "\n\n";
//     // std::cout << "Final batch y:\n" << y << "\n\n";
//     // std::cout << "Final loss: " << loss.item<double>() << "\n";
//     // std::cout << "\nTraining time: " << elapsed << " ms\n";

//     return 0;
// }

// #include <iostream>
// #include <vector>
// #include <array>
// #include <pybind11/embed.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <functional>
// #include <memory>
// #include <chrono>
// #include <torch/torch.h>
// // #include <autodiff/forward/dual.hpp>
// // #include <autodiff/forward/dual/eigen.hpp>
// // #include <autodiff/reverse/var.hpp>
// // #include <autodiff/reverse/var/eigen.hpp>
// // #include "FNN.hpp"
// #include "FNN.hpp"
// #include "boundary_condition/Dirichlet_BC.hpp"
// #include "operator/Differential_Operators.hpp"
// #include "Losses.hpp"
// // #include "optimizer/Gradient_Descent.hpp"
// // #include "optimizer/ParaflowS.hpp"
// #include "Pde.hpp"
// // #include "Model.hpp"

// namespace py = pybind11;
// // using namespace autodiff;

// int main(){
//     py::scoped_interpreter guard{}; // start the interpreter and keep it alive
    
//     py::module_ os = py::module_::import("os");
//     os.attr("environ")["DDE_BACKEND"] = "pytorch";

//     // Optional: inspect Python path if needed
//     // py::module_ sys = py::module_::import("sys");

//     // Import deepxde after backend is set
//     py::module_ dde = py::module_::import("deepxde");

//     // Set random seed for reproducibility
//     dde.attr("config").attr("set_random_seed")(123); 
     
//     // // py::module_ problem_settings = py::module_::import("problem_settings"); // import problem informations like domain, pde, bc,...

//     // geometry
//     std::vector<std::array<double, 2>> vertices = {{0.0, 0.0}, {1.0, 0.0}, {1.0, -1.0}, {-1.0, -1.0}, {-1.0, 1.0}, {0.0, 1.0}};

//     py::object geom = dde.attr("geometry").attr("Polygon")(vertices);

//     // boundary conditions (Torch-based)
//     std::function<bool(const tensor&, bool)> on_boundary = [](const tensor&, bool on_bc){ return on_bc; };

//     // Dirichlet target: zero function, returns (N,1) tensor of zeros
//     std::function<tensor(const tensor&)> func = [](const tensor& x){
//         return torch::zeros({x.size(0), 1}, x.options());
//     };

//     std::vector<std::shared_ptr<Boundary_Condition>> bc_vector;
//     bc_vector.push_back(std::make_shared<Dirichlet_BC>(geom, func, on_boundary));

//     // Simple network (needed to build an autograd graph for the PDE residual)
//     FNN<Activation::Tanh> net(std::vector<int64_t>{2, 50, 50, 50, 50, 1});
//     net.to(torch::kFloat32);
//     net.initialize<Initializer_weight::Glorot_Uniform, Initializer_bias::Constant>(0.0);

//     std::function<std::vector<tensor>(const tensor&, const tensor&)> pde_equation =
//         [](const tensor& inputs, const tensor& outputs) {
//             tensor residual = - differential_operators::laplacian(outputs, inputs, /*component=*/0) -1;
//             return std::vector<tensor>{residual};
//         };

//     Pde pde(geom, pde_equation, bc_vector, /*Num_domain=*/1200, /*Num_boundary=*/120, /*Num_test=*/150);

//     // -----------------------------
//     // PyTorch Gradient Descent (SGD)
//     // -----------------------------
//     const double lr = 0.001;
//     const int iterations = 200;
//     const int batch_size = 0;
//     const double w_pde = 1.0;
//     const double w_bc = 1.0;
//     int budget = 1e5;
//     int b_used = 0;

//     torch::optim::SGD optimizer(net.parameters(), torch::optim::SGDOptions(lr));

//     std::function<tensor(const tensor& r)> loss_fn = losses::get_loss("MSE");
//     // After dataset/geometry construction, we can release the GIL for the whole
//     // training loop (no Python calls are needed inside the loop).
//     py::gil_scoped_release no_gil;

//     auto t0 = std::chrono::high_resolution_clock::now();

//     for (int it = 0; it < iterations; ++it) {
//         tensor& batch_x = pde.train_next_batch(batch_size);
//         const int64_t N = batch_x.size(0);
//         b_used += batch_x.size(0);

//         // Leaf input for autograd w.r.t. x (needed for Laplacian)
//         tensor x = batch_x.clone().detach().set_requires_grad(true);

//         // Forward
//         tensor u = net.forward(x); // [N,1]

//         // PDE residual on full batch, then split BC/PDE by convention
//         const int total_bc = pde.total_bc_in_last_batch();
//         TORCH_CHECK(total_bc >= 0 && total_bc <= N, "Invalid total_bc in batch");

//         tensor residual_full = pde_equation(x, u).at(0); // [N,1]

//         tensor pde_loss = torch::zeros({}, u.options());
//         const int64_t pde_rows = N - total_bc;
//         if (pde_rows > 0) {
//             tensor r_pde = residual_full.narrow(0, total_bc, pde_rows);
//             pde_loss = loss_fn(r_pde);//.square().mean();
//         }

//         tensor bc_loss = torch::zeros({}, u.options());
//         if (total_bc > 0) {
//             // The batch layout is [BC points | PDE points], so BC range is [0, total_bc).
//             tensor bc_err = bc_vector[0]->error(batch_x, batch_x, u, 0, total_bc);
//             bc_loss = loss_fn(bc_err);//.square().mean();
//         }

//         tensor loss = w_pde * pde_loss + w_bc * bc_loss;

//         optimizer.zero_grad();
//         loss.backward();
//         optimizer.step();

//         if (it % 10 == 0) {
//             std::cout << "it=" << it
//                       << " loss=" << loss.item<double>()
//                       << " pde=" << pde_loss.item<double>()
//                       << " bc=" << bc_loss.item<double>()
//                       << "\n";
//         }
//         if (b_used >= budget){
//             std::cout << it ;
//             break;
//         }
//     }

//     auto t1 = std::chrono::high_resolution_clock::now();
//     auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
//     std::cout << "Training time: " << ms << " ms\n";

//     // // collect all pde data into a single class
//     // py::object data = dde.attr("data").attr("PDE")(geom, problem_settings.attr("pde"), bc_list, 16, 2, py::arg("solution") = problem_settings.attr("func_ex"), py::arg("num_test") = 100);

//     // initilize the FNN class
//     // std::srand(42);//(unsigned int)) time(0));

//     // torch::manual_seed(0);

//     // std::vector<int64_t> layer_size = {2,32,32,1};

//     // constexpr Initializer_bias Ib = Initializer_bias::Constant;
//     // constexpr Initializer_weight Iw = Initializer_weight::He_Norm;
//     // constexpr Activation A = Activation::Tanh;

//     // FNN<A> net(layer_size);
//     // net.to(torch::kDouble);

//     // net.initialize<Iw, Ib>();

//     // net.print();

//     // // Optimizer
//     // torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));

//     // const int64_t epochs = 500;
//     // torch::Tensor x, y, target, loss;

//     // auto t_start = std::chrono::high_resolution_clock::now();

//     // for(int64_t epoch = 0; epoch < epochs; ++epoch){
//     //     // Input batch
//     //     x = torch::rand({5, 2}, torch::dtype(torch::kDouble));

//     //     // Dummy target for demonstration (here: all ones)
//     //     target = torch::ones({5, 1}, torch::dtype(torch::kDouble));

//     //     // Forward pass
//     //     y = net.forward(x);

//     //     // Compute loss (Mean Squared Error)
//     //     loss = torch::mse_loss(y, target);

//     //     // Backward pass
//     //     optimizer.zero_grad();
//     //     loss.backward();
//     //     optimizer.step();

//     //     if(epoch % 50 == 0){
//     //         std::cout << "Epoch " << epoch
//     //                   << ", loss = " << loss.item<double>() << '\n';
//     //     }
//     // }

//     // auto t_end = std::chrono::high_resolution_clock::now();
//     // auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

//     // std::cout << "\nFinal batch x:\n" << x << "\n\n";
//     // std::cout << "Final batch y:\n" << y << "\n\n";
//     // std::cout << "Final loss: " << loss.item<double>() << "\n";
//     // std::cout << "\nTraining time: " << elapsed << " ms\n";

//     return 0;
// }