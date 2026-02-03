#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

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
    Gradient_Descent(std::shared_ptr<Pde> data,
                     std::shared_ptr<NetT> net,
                     double lr,
                     std::function<tensor(const tensor&)> loss_fn)
        : Optimizer<NetT>(std::move(data), std::move(net), std::move(loss_fn)), lr(lr),
          operator_(this->net->parameters(), torch::optim::SGDOptions(lr)) {}

    virtual ~Gradient_Descent() = default;

    Result train(int batch_size, int budget, int max_iterations, bool verbose) override;
};

template <class NetT>
Result Gradient_Descent<NetT>::train(int batch_size, int budget, int max_iterations, bool verbose){
    if(verbose){
        std::cout << "Starting optimization with Gradient Descent..." << std::endl;
    }
    // Training is fully in C++/torch; release GIL to avoid blocking embedded Python.
    py::gil_scoped_release no_gil;

    this->budget_used = 0;
    Result res;

    int disp_every = budget / 100;
    
    tensor x;
    tensor u;
    tensor loss;

    tensor test;
    tensor u_test;
    tensor loss_test;
    if (this->use_test) {
        test = this->data->get_test().detach().clone().set_requires_grad(true);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < max_iterations; ++it) {
        tensor& batch_x = this->data->train_next_batch(batch_size);

        // necessary line to compute the differential operator w.r.t. the input of the PINN
        x = batch_x.clone().detach().set_requires_grad(true);

        u = this->net->forward(x);

        const std::vector<tensor> losses = this->data->losses(x, u, this->loss_fn);
        loss = torch::stack(losses).sum();

        operator_.zero_grad();
        loss.backward();
        operator_.step();

        this->budget_used += batch_x.size(0);

        if (verbose && (it % disp_every == 0)) {
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