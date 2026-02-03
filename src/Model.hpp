#ifndef MODEL_HPP
#define MODEL_HPP

/**
 * @file Model.hpp
 * @brief DeepXDE-like Model wrapper for the C++/LibTorch PINN pipeline.
 *
 * @details
 * `Model` is a thin façade that:
 * - stores a `Pde` dataset and a network instance
 * - creates a concrete optimizer in `compile()`
 * - delegates training to the optimizer in `train()`
 * - provides a `save_data()` utility mirroring the Python implementation
 */

#include <memory>
#include <utility>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <torch/torch.h>

#include "pde/Pde.hpp"
#include "Losses.hpp"
#include "optimizer/Optimizer.hpp"
#include "optimizer/Gradient_Descent.hpp"
#include "optimizer/ParaflowS.hpp"

/**
 * @details
 * Minimal "Model" wrapper (in the spirit of DeepXDE's `Model`): it holds
 * the data object, the neural network, and an optimizer/trainer.
 *
 * Design choice: `Model::train()` is a thin wrapper that delegates to the
 * real training loop implemented in `Optimizer::train(...)`.
 *
 * The PDE residual itself remains a callable (typically a lambda in `main.cpp`)
 * stored inside `Pde`.
 */
template <class NetT>
class Model {
private:
    std::shared_ptr<Pde> data;
    std::shared_ptr<NetT> net;

    /** @brief Compile-time settings (similar to DeepXDE). */
    std::string opt_name;
    double lr;
    std::string loss_name;

    /**
     * @brief Stored for possible inspection/debug.
     * @details The optimizer also keeps its own copy.
     */
    std::function<tensor(const tensor&)> loss_fn;

    /**
     * @brief Project optimizer interface created in `compile()`.
     * @details The concrete type may also wrap a `torch::optim::*` instance.
     */
    std::shared_ptr<Optimizer<NetT>> opt;

public:
        /**
         * @brief Construct a model.
         * @param data Shared PDE dataset/geometry.
         * @param net Shared neural network.
         */
    Model(std::shared_ptr<Pde> data,
          std::shared_ptr<NetT> net)
        : data(std::move(data)), net(std::move(net)) {}

    Pde& get_data() { return *data; }
    const Pde& get_data() const { return *data; }

    NetT& get_net() { return *net; }
    const NetT& get_net() const { return *net; }

    const std::string& get_opt_name() const { return opt_name; }

    /**
     * @brief Compile the model with an optimizer that does not require `n_fine`.
     *
     * @param optimizer_name Optimizer identifier (e.g. "SGD", "Gradient_Descent").
     * @param lr Learning rate.
     * @param loss Loss reducer identifier (e.g. "MSE").
     * @param verbose Print compilation information.
     */
    void compile(
        const std::string& optimizer_name, double lr, const std::string& loss = "MSE", bool verbose = true) {
        opt_name = optimizer_name;
        this->lr = lr;
        loss_name = loss;

        this->loss_fn = losses::get_loss(loss);

        /** @details Instantiate the requested optimizer implementation. */
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

    /**
     * @brief Compile the model with an optimizer that requires `n_fine`.
     *
     * @param optimizer_name Optimizer identifier (currently "ParaFlowS").
     * @param lr Learning rate.
     * @param n_fine Number of fine steps per outer iteration.
     * @param loss Loss reducer identifier (e.g. "MSE").
     * @param verbose Print compilation information.
     */
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

    /**
     * @brief Train the compiled model.
     *
     * @param iterations Maximum iterations.
     * @param batch_size Batch size; 0 means full batch.
     * @param budget Maximum number of samples to consume (0 disables budget stop).
     * @param verbose Enable optimizer logging.
     * @return Result Training summary.
     */
    Result train(int iterations, int batch_size, int budget, bool verbose = false) {
        if (!opt) {
            throw std::runtime_error("Model::train: call compile() before train()");
        }
        return opt->train(batch_size, budget, iterations, verbose);
    }

    /**
     * @brief Plot training (and optionally test) loss history.
     *
     * @details
     * Delegates to the compiled optimizer's `plot_cost_history()`.
     * The plot is produced via embedded Python/matplotlib (so it acquires the GIL).
     *
     * Call this after `train()` if you want a visualization of the logged history.
     */
    void plot_loss_history() const{
        if (!opt) {
            throw std::runtime_error("Model::plot_loss_history: call compile() before plot_loss_history()");
        }
        return opt->plot_cost_history();
    }
    
    /**
     * @brief Append a training run row to a CSV file (Python-compatible schema).
     *
     * @details
     * Mirrors the Python `Model.save_data(...)` helper.
     * CSV schema:
     * `optimizer_name,batch_size,lr,final_budget,budget,n_fine,final_loss,epochs,time_train`
     *
     * The `final_budget` field follows your Python convention: the remaining
     * budget after training ends (`budget - budget_used`).
     *
     * @param filename Output CSV path.
     * @param result Result struct returned by the optimizer.
     */
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
};

#endif