#ifndef MODEL_HPP
#define MODEL_HPP

#include <memory>
#include <utility>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <torch/torch.h>

#include "Pde.hpp"
#include "Losses.hpp"
#include "optimizer/Optimizer.hpp"
#include "optimizer/Gradient_Descent.hpp"
#include "optimizer/ParaflowS.hpp"

// Minimal "Model" wrapper (in the spirit of DeepXDE's Model): it holds
// the data object, the neural network, and an optimizer/trainer.
//
// Design choice: Model::train() is just a thin wrapper that calls the
// real training loop implemented in optimizer_->train(...).
//
// The PDE itself remains a lambda in main.cpp and is stored inside Pde.
template <class NetT>
class Model {
private:
    std::shared_ptr<Pde> data;
    std::shared_ptr<NetT> net;

    // Compile-time settings (similar to DeepXDE)
    std::string opt_name;
    double lr;
    std::string loss_name;


    // Stored for possible inspection/debug; the optimizer also keeps its own copy.
    std::function<tensor(const tensor&)> loss_fn;

    // Project optimizer interface (created at compile time).
    // Concrete type also inherits from torch::optim::*.
    std::shared_ptr<Optimizer<NetT>> opt;

public:
    Model(std::shared_ptr<Pde> data,
          std::shared_ptr<NetT> net)
        : data(std::move(data)), net(std::move(net)) {}

    Pde& get_data() { return *data; }
    const Pde& get_data() const { return *data; }

    NetT& get_net() { return *net; }
    const NetT& get_net() const { return *net; }

    const std::string& get_opt_name() const { return opt_name; }

    void compile(
        const std::string& optimizer_name, double lr, const std::string& loss = "MSE", bool verbose = true) {
        opt_name = optimizer_name;
        this->lr = lr;
        loss_name = loss;

        this->loss_fn = losses::get_loss(loss);

        // Optimizer
        if (optimizer_name == "ParaFlowS"){
            throw std::runtime_error("Model::compile: ParaFlowS requires n_fine. Use compile(optimizer, lr, n_fine, ...)");
        }
        if (optimizer_name == "SGD" || optimizer_name == "Gradient_Descent") {
            opt = std::make_shared<Gradient_Descent<NetT>>(data, net, this->lr, this->loss_fn);

            if (verbose){
                std::cout << "Compilation done " << std::endl;
            }
            return;
        }

        throw std::runtime_error("Optimizer not implemented: '" + optimizer_name + "'");

    }

    void compile(
        const std::string& optimizer_name, double lr, int n_fine, const std::string& loss = "MSE", bool verbose = true) {
        opt_name = optimizer_name;
        this->lr = lr;
        loss_name = loss;

        this->loss_fn = losses::get_loss(loss);

        if (optimizer_name == "ParaFlowS") {
            opt = std::make_shared<ParaflowS<NetT>>(data, net, this->lr, n_fine, this->loss_fn);

            if (verbose) {
                std::cout << "Compilation done " << std::endl;
            }
            return;
        }

        throw std::runtime_error("Model::compile: ParaFlowS is the only optimizer that requires n_fine. Use compile(optimizer, lr, ...)");
    }

    Result train(int iterations, int batch_size, int budget, bool verbose = false) {
        if (!opt) {
            throw std::runtime_error("Model::train: call compile() before train()");
        }
        return opt->train(batch_size, budget, iterations, verbose);
    }
    
    // Mirrors the Python Model.save_data(...) helper used in your DeepXDE scripts.
    // CSV schema:
    // optimizer_name,batch_size,lr,final_budget,budget,n_fine,final_loss,epochs,time_train,optimizer_counter
    void save_data(
        const std::filesystem::path& filename,
        const Result& result) const {

        std::error_code ec;
        auto parent = filename.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent, ec);
        }

        const bool has_header = std::filesystem::exists(filename, ec) &&
                                std::filesystem::is_regular_file(filename, ec) &&
                                std::filesystem::file_size(filename, ec) > 0;

        std::ofstream out(filename, std::ios::app);
        if (!out) {
            throw std::runtime_error("Model::save_data: failed to open CSV file: " + filename.string());
        }

        if (!has_header) {
            out << "optimizer_name,batch_size,lr,final_budget,budget,n_fine,final_loss,epochs,time_train\n";
        }

        const int final_budget = result.budget - result.budget_used;
        const double time_train_s = result.total_time_ms / 1000.0;


        out << opt_name << ','
            << result.batch_size << ','
            << std::setprecision(12) << result.lr << ','
            << final_budget << ','
            << result.budget << ','
            << result.n_fine << ','
            << std::setprecision(12) << result.final_loss << ','
            << result.epoch << ','
            << std::setprecision(12) << time_train_s
            << '\n';
    }

    // void compile(
    //     const std::string& optimizer_name, double lr, int n_fine, const std::string& loss = "MSE", int verbose = 1) {
    //     opt_name = optimizer_name;
    //     this->lr = lr;
    //     loss_name = loss;

    //     std::function<tensor(const tensor&)> loss_fn = losses::get_loss(loss);

    //     // Optimizer
    //     if (optimizer_name != "ParaFlowS"){
    //         throw std::runtime_error("Only the ParaFlowS optimizer uses n_fine passes, your optimizer is " + optimizer_name);
    //     }

    //     throw std::runtime_error("ParaFlowS is not wired to the torch PINN pipeline yet.");

    // }

    // // Simple training loop using torch autograd + Pde::losses.
    // // Sums all loss terms; if loss_weights were provided in compile(), applies them.
    // void train(int iterations, int batch_size = 0) {
    //     if (!opt_) {
    //         throw std::runtime_error("Model::train: call compile() before train()");
    //     }
    //     if (!loss_fn_) {
    //         throw std::runtime_error("Model::train: loss function not initialized (call compile())");
    //     }

    //     // Delegate the training loop to the optimizer/trainer.
    //     opt_->train(*data_, *net_, loss_fn_, iterations, batch_size, loss_weights_, verbose_);
    // }
};

#endif