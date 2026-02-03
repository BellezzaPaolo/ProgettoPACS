#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

/**
 * @file Gradient_Descent.hpp
 * @brief SGD-based optimizer/trainer for the PINN model.
 *
 * @details
 * This optimizer performs standard stochastic gradient descent using
 * `torch::optim::SGD` over the network parameters.
 *
 * Important for PINNs: the input batch `x` is cloned/detached and then
 * `requires_grad(true)` is enabled so that autograd can compute PDE
 * derivatives (e.g. Laplacian) with respect to the inputs.
 */

#include <pybind11/embed.h>
#include <chrono>
#include <iostream>

#include "Optimizer.hpp"

namespace py = pybind11;

using tensor = torch::Tensor;

template <class NetT>
class Gradient_Descent : public Optimizer<NetT> {
private:
    double lr;
    torch::optim::SGD operator_;
public:
    /**
     * @brief Construct a Gradient Descent optimizer.
     * @param data Dataset/geometry wrapper.
     * @param net Network to optimize.
     * @param lr Learning rate.
     * @param loss_fn Scalar loss reducer (e.g. MSE).
     */
    Gradient_Descent(std::shared_ptr<Pde> data,
                     std::shared_ptr<NetT> net,
                     double lr,
                     std::function<tensor(const tensor&)> loss_fn)
        : Optimizer<NetT>(std::move(data), std::move(net), std::move(loss_fn)), lr(lr),
          operator_(this->net->parameters(), torch::optim::SGDOptions(lr)) {}

    virtual ~Gradient_Descent() = default;

    /**
     * @brief Train using SGD.
     *
     * @param batch_size Batch size; 0 means full batch.
     * @param budget Maximum number of samples to consume (0 disables budget stop).
     * @param max_iterations Maximum optimizer iterations.
     * @param verbose Enable periodic logging and loss history tracking.
     * @return Result Training summary.
     */
    Result train(int batch_size, int budget, int max_iterations, bool verbose) override;
};

template <class NetT>
Result Gradient_Descent<NetT>::train(int batch_size, int budget, int max_iterations, bool verbose){
    /**
     * @details
     * Training loop outline:
     * 1) Fetch a batch from `Pde::train_next_batch()`.
     * 2) Build a leaf tensor `x` with `requires_grad(true)` to enable
     *    autograd-based PDE operators inside `Pde::losses()`.
     * 3) Forward pass, compute physics/data losses, backward, SGD step.
     * 4) Track consumed samples via `budget_used` and stop if `budget` is reached.
     *
     * Notes:
     * - `batch_size == 0` means full batch.
     * - `budget == 0` disables the budget stop condition.
     */
    if(verbose){
        std::cout << "Starting optimization with Gradient Descent..." << std::endl;
    }
    // Training is fully in C++/torch; release GIL to avoid blocking embedded Python.
    py::gil_scoped_release no_gil;

    this->budget_used = 0;
    Result res;
    
    tensor x;
    tensor u;
    tensor loss;

    tensor test;
    tensor u_test;
    tensor loss_test;
    if (this->use_test) {
        // Keep a leaf tensor with gradients enabled: test loss may need PDE derivatives w.r.t. x.
        test = this->data->get_test().detach().clone().set_requires_grad(true);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < max_iterations; ++it) {
        tensor& batch_x = this->data->train_next_batch(batch_size);

        // Create a leaf input tensor with gradients enabled for PINN differential operators.
        x = batch_x.clone().detach().set_requires_grad(true);

        u = this->net->forward(x);

        const std::vector<tensor> losses = this->data->losses(x, u, this->loss_fn);
        loss = torch::stack(losses).sum();

        operator_.zero_grad();
        loss.backward();
        operator_.step();

        this->budget_used += batch_x.size(0);

        if (verbose && (it % 10 == 0)) {
            this->loss_history_train.push_back(loss.item<double>());

            if (this->use_test) {
                u_test = this->net->forward(test);
                loss_test = torch::stack(this->data->losses(test, u_test, this->loss_fn)).sum();
                this->loss_history_test.push_back(loss_test.item<double>());

                std::cout << "it =" << it
                          << " loss =" << loss.item<double>()
                          << " loss_test =" << loss_test.item<double>()
                          << std::endl;
            } 
            else {
                std::cout << "it =" << it << " loss =" << loss.item<double>() << std::endl;
            }
        }

        if (budget > 0 && this->budget_used >= budget) {
            res.epoch = it + 1;
            break;
        }

        res.epoch = it;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Final loss on full training set
    tensor& batch_x = this->data->train_next_batch(0);
    x = batch_x.detach().clone().set_requires_grad(true);
    u = this->net->forward(x);
    loss = torch::stack(this->data->losses(x, u, this->loss_fn)).sum();
    res.final_loss = (loss).item<double>();

    res.total_time_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
    res.budget_used = this->budget_used;
    res.budget = budget;
    if (batch_size == 0){
        res.batch_size = batch_x.size(0);
    }
    else{
        res.batch_size = batch_size;
    }
    res.lr = this->lr;
    return res;
}

#endif