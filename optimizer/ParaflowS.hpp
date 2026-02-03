#ifndef PARAFLOWS_HPP
#define PARAFLOWS_HPP

#include <chrono>
#include <iostream>
#include <vector>

#include <torch/torch.h>
#include "Optimizer.hpp"

/**
 * @brief ParaFlowS optimizer/trainer for the torch PINN pipeline.
 *
 * @details
 * This implements the ParaFlowS scheme as in the reference Python version in
 * `python_implementation/paraflow.py`:
 * - A *coarse* operator computes a large-step candidate (stored internally).
 * - A *fine* operator performs `n_fine` small GD steps (updates parameters).
 * - A correction loop runs up to `n_coarse` coarse steps while enforcing
 *   monotonic decrease w.r.t. the fine loss.
 *
 * The interface matches the other project optimizers: it derives from
 * `Optimizer<NetT>` and implements `train()`.
 */
template <class NetT>
class ParaflowS final : public Optimizer<NetT> {
private:
    double lr_fine;
    int n_fine;
    int n_coarse;
    double lr_coarse;

    // Use torch optimizers for update rules (future-proof: can be swapped to Adam, etc.).
    torch::optim::SGD coarse_opt_;
    torch::optim::SGD fine_opt_;

    // Per-parameter state (aligned with net->parameters())
    std::vector<tensor> bff1;
    std::vector<tensor> correction;
    std::vector<tensor> params_old;
    std::vector<tensor> params_backup;

    // tensor compute_train_loss_and_backward(int batch_size) {
    //     tensor& batch_x_ref = this->data->train_next_batch(batch_size);
    //     tensor x = batch_x_ref.detach().clone().set_requires_grad(true);

    //     this->net->zero_grad();
    //     tensor u = this->net->forward(x);
    //     const std::vector<tensor> losses = this->data->losses(x, u, this->loss_fn);
    //     tensor loss = torch::stack(losses).sum();
    //     loss.backward();

    //     this->budget_used += static_cast<int>(batch_x_ref.size(0));
    //     return loss;
    // }

    // tensor compute_test_loss() {
    //     const tensor& test = this->data->get_test();
    //     if (!test.defined() || test.numel() == 0) {
    //         return {};
    //     }

    //     tensor x = test.detach().clone().set_requires_grad(true);
    //     this->net->zero_grad();
    //     tensor u = this->net->forward(x);
    //     return torch::stack(this->data->losses(x, u, this->loss_fn)).sum();
    // }

    void initialize_state() {
        const auto params = this->net->parameters();
        if (!bff1.empty() && bff1.size() == params.size()) {
            return;
        }

        bff1.clear();
        correction.clear();
        params_old.clear();
        params_backup.clear();

        bff1.reserve(params.size());
        correction.reserve(params.size());
        params_old.reserve(params.size());
        params_backup.reserve(params.size());

        for (const auto& p : params) {
            // Detached clones to avoid graph tracking and aliasing.
            bff1.push_back(torch::zeros_like(p).detach());
            correction.push_back(torch::zeros_like(p).detach());
            params_old.push_back(torch::zeros_like(p).detach());
            params_backup.push_back(torch::zeros_like(p).detach());
        }
    }

    // void set_params_from_state_sum(const std::vector<tensor>& a, const std::vector<tensor>& b) {
    //     auto params = this->net->parameters();
    //     torch::NoGradGuard no_grad;
    //     for (size_t i = 0; i < params.size(); ++i) {
    //         params[i].copy_(a[i] + b[i]);
    //     }
    // }

    void copy_params_to_state(std::vector<tensor>& dst) {
        const auto params = this->net->parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < params.size(); ++i) {
            dst[i].copy_(params[i].detach());
        }
    }

    double coarse_solver(int batch_size) {
        tensor& batch_x_ref = this->data->train_next_batch(batch_size);
        tensor x = batch_x_ref.detach().clone().set_requires_grad(true);

        this->net->zero_grad();
        tensor u = this->net->forward(x);
        const std::vector<tensor> losses = this->data->losses(x, u, this->loss_fn);
        tensor loss = torch::stack(losses).sum();
        loss.backward();

        this->budget_used += static_cast<int>(batch_x_ref.size(0));

        // Build the coarse candidate without permanently updating the live parameters:
        // 1) backup current params
        // 2) take one coarse optimizer step
        // 3) store the stepped params into bff1
        // 4) restore original params (so fine solver starts from pre-coarse state)
        {
            auto params = this->net->parameters();
            torch::NoGradGuard no_grad;
            for (size_t i = 0; i < params.size(); ++i) {
                params_backup[i].copy_(params[i].detach());
            }
        }

        coarse_opt_.step();

        {
            auto params = this->net->parameters();
            torch::NoGradGuard no_grad;
            for (size_t i = 0; i < params.size(); ++i) {
                bff1[i].copy_(params[i].detach());
                params[i].copy_(params_backup[i]);
            }
        }

        return loss.item<double>();
    }

    double fine_solver(int batch_size, int budget) {
        tensor loss;
        tensor x;
        tensor u;
        // std::vector<tensor> losses;

        for (int k = 0; k < n_fine; ++k) {
            tensor& batch_x_ref = this->data->train_next_batch(batch_size);
            x = batch_x_ref.clone().detach().set_requires_grad(true);

            u = this->net->forward(x);
            const std::vector<tensor> losses = this->data->losses(x, u, this->loss_fn);
            loss = torch::stack(losses).sum();

            // std::cout << "loss_fine "<< loss.item<double>() << std::endl;
            
            this->net->zero_grad();
            loss.backward();

            fine_opt_.step();

            this->budget_used += static_cast<int>(batch_x_ref.size(0));

            if (budget > 0 && this->budget_used >= budget) {
                break;
            }
        }

        return loss.item<double>();
    }

public:
    ParaflowS(std::shared_ptr<Pde> data,
              std::shared_ptr<NetT> net,
              double lr_fine,
              int n_fine,
              std::function<tensor(const tensor&)> loss_fn,
              int n_coarse = 200)
        : Optimizer<NetT>(std::move(data), std::move(net), std::move(loss_fn)),
          lr_fine(lr_fine),
          n_fine(n_fine),
          n_coarse(n_coarse),
          lr_coarse(lr_fine * static_cast<double>(n_fine)),
          coarse_opt_(this->net->parameters(), torch::optim::SGDOptions(lr_coarse)),
          fine_opt_(this->net->parameters(), torch::optim::SGDOptions(lr_fine)) {
        if (n_fine <= 0) {
            throw std::runtime_error("ParaflowS: n_fine must be > 0");
        }
        if (n_coarse <= 0) {
            throw std::runtime_error("ParaflowS: n_coarse must be > 0");
        }
        initialize_state();
    }

    Result train(int batch_size, int budget, int max_iterations, bool verbose) override {
        if (verbose){
            std::cout << "Starting optimization with ParaFlowS..." << std::endl;
        }
        // Training is fully in C++/torch; release GIL to avoid blocking embedded Python.
        py::gil_scoped_release no_gil;

        initialize_state();
        this->budget_used = 0;
        bool stay = true;

        Result res;
        double loss_fine = 0;
        double loss_coarse = 0;

        tensor test;
        tensor x_test;
        tensor u_test;
        double loss_test;
        tensor loss_test_t;
        if (this->use_test){
            // Keep a local leaf copy with gradients enabled (used for PDE derivatives in test loss).
            test = this->data->get_test().detach().clone().set_requires_grad(true);
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int it = 0; it < max_iterations; ++it) {
            // 1) coarse pass (compute bff1)
            loss_coarse = coarse_solver(/*batch_size=*/0);
            // std::cout << "loss_coarse " << loss_coarse << std::endl;

            if (budget > 0 && this->budget_used >= budget) {
                res.epoch = it + 1;
                break;
            }

            // 2) fine pass (updates params)
            loss_fine = fine_solver(batch_size, budget);
            // std::cout << "loss_fine " << loss_fine << std::endl;

            // 3) correction loop
            stay = true;
            loss_coarse = loss_fine;

            {
                const auto params = this->net->parameters();
                torch::NoGradGuard no_grad;
                for (size_t i = 0; i < params.size(); ++i) {
                    correction[i].copy_(params[i] - bff1[i]);
                }
            }

            for (int j = 0; j < n_coarse; ++j) {
                // params = bff1 + correction
                {
                    auto params = this->net->parameters();
                    torch::NoGradGuard no_grad;
                    for (size_t i = 0; i < params.size(); ++i) {
                        params[i].copy_(bff1[i] + correction[i]);
                    }
                }

                if (budget > 0 && this->budget_used >= budget) {
                    res.epoch = it;
                    stay = false;
                    break;
                }

                loss_coarse = coarse_solver(/*batch_size=*/0);
                // std::cout << "loss coarse = " << loss_coarse << std::endl;

                if (loss_coarse <= loss_fine || j == 0) {
                    // save iteration
                    {
                        const auto params = this->net->parameters();
                        torch::NoGradGuard no_grad;
                        for (size_t i = 0; i < params.size(); ++i) {
                            params_old[i].copy_(params[i].detach());
                        }
                    }
                    loss_fine = loss_coarse;
                } else {
                    // revert to best accepted coarse state
                    {
                        auto params = this->net->parameters();
                        torch::NoGradGuard no_grad;
                        for (size_t i = 0; i < params.size(); ++i) {
                            params[i].copy_(params_old[i]);
                        }
                    }
                    stay = false;
                    break;
                }
            }

            if (verbose && (it % 1 == 0)) {
                this->loss_history_train.push_back(loss_fine);

                if(this->use_test){

                    x_test = test.detach().clone().set_requires_grad(true);
                    this->net->zero_grad();
                    u_test = this->net->forward(x_test);
                    loss_test_t = torch::stack(this->data->losses(x_test, u_test, this->loss_fn)).sum();
                    loss_test = loss_test_t.item<double>();
                    this->loss_history_test.push_back(loss_test);
                    std::cout << "it =" << it
                            << " loss =" << loss_fine
                            << " loss_test =" << loss_test
                            << " lr = " << this -> lr_fine
                            << " n_fine = " << this-> n_fine
                            << std::endl;

                }
                else {
                    std::cout << "it = " << it << " loss = " << loss_fine << std::endl;
                }
            }

            if (budget > 0 && this->budget_used >= budget) {
                res.epoch = it + 1;
                break;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        // Final loss on full training set
        tensor& full_x_ref = this->data->train_next_batch(0);
        tensor x_full = full_x_ref.detach().clone().set_requires_grad(true);
        this->net->zero_grad();
        tensor u_full = this->net->forward(x_full);
        tensor final_loss = torch::stack(this->data->losses(x_full, u_full, this->loss_fn)).sum();
        res.final_loss = final_loss.item<double>();

        res.total_time_ms = static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
        res.budget_used = this->budget_used;
        res.budget = budget;
        if (batch_size == 0){
            res.batch_size = full_x_ref.size(0);
        }
        else{
            res.batch_size = batch_size;
        }
        res.n_fine = this->n_fine;
        res.lr = this->lr_fine;
        return res;
    }

};

#endif