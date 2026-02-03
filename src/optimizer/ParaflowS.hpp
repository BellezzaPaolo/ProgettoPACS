#ifndef PARAFLOWS_HPP
#define PARAFLOWS_HPP

/**
 * @file ParaflowS.hpp
 * @brief ParaFlowS optimizer/trainer (coarse/fine + correction) for PINNs.
 *
 * @details
 * ParaFlowS alternates between:
 * - a *coarse* solver that builds a candidate parameter state (stored in `bff1`)
 * - a *fine* solver that performs `n_fine` standard SGD steps on the live network
 * - a correction loop that combines coarse candidate + correction and enforces
 *   monotonicity with respect to the fine loss.
 *
 * For PINNs, input batches are created as leaf tensors with
 * `requires_grad=true` to enable autograd-based differential operators.
 */

#include <chrono>
#include <iostream>
#include <vector>

#include <torch/torch.h>
#include "Optimizer.hpp"

/**
 * @brief ParaFlowS optimizer/trainer for the torch PINN pipeline.
 *
 * @details
 * This implements the ParaFlowS scheme:
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

    /** @brief Torch optimizers implementing the update rules (coarse and fine). */
    torch::optim::SGD coarse_opt_;
    torch::optim::SGD fine_opt_;

    /** @brief Per-parameter state buffers (aligned with `net->parameters()`). */
    std::vector<tensor> bff1;
    std::vector<tensor> correction;
    std::vector<tensor> params_old;
    std::vector<tensor> params_backup;

    /**
     * @brief Allocate (or reallocate) internal per-parameter buffers.
     *
     * @details
     * ParaFlowS keeps several parameter-shaped vectors:
     * - `bff1`: coarse candidate parameters
     * - `correction`: fine - coarse correction term
     * - `params_old`: last accepted parameters in the monotonicity loop
     * - `params_backup`: temporary storage to restore parameters after a coarse step
     *
     * Buffers are resized only when the number of network parameters changes.
     */
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
            /** @details Detached buffers avoid graph tracking and aliasing. */
            bff1.push_back(torch::zeros_like(p).detach());
            correction.push_back(torch::zeros_like(p).detach());
            params_old.push_back(torch::zeros_like(p).detach());
            params_backup.push_back(torch::zeros_like(p).detach());
        }
    }

    /**
     * @brief Run one coarse step and store the candidate parameters in `bff1`.
     *
     * @param batch_size Batch size to use for the coarse loss/gradient evaluation.
     *                  Use 0 for full-batch.
     * @return Coarse loss value (scalar) before the coarse update.
     *
     * @details
     * This function:
     * 1) Builds a leaf input tensor `x` with `requires_grad(true)` for PINN operators.
     * 2) Computes loss and gradients.
     * 3) Backs up current parameters.
     * 4) Applies `coarse_opt_` step to the live network.
     * 5) Copies the stepped parameters into `bff1` (coarse candidate), then restores
     *    the original parameters from `params_backup`.
     *
     * The backup/restore ensures coarse steps do not permanently perturb the live
     * parameters before fine/correction logic decides what to accept.
     */
    double coarse_solver(int batch_size) {
        tensor& batch_x_ref = this->data->train_next_batch(batch_size);
        tensor x = batch_x_ref.detach().clone().set_requires_grad(true);

        this->net->zero_grad();
        tensor u = this->net->forward(x);
        const std::vector<tensor> losses = this->data->losses(x, u, this->loss_fn);
        tensor loss = torch::stack(losses).sum();
        loss.backward();

        this->budget_used += static_cast<int>(batch_x_ref.size(0));

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

    /**
     * @brief Run the fine solver for up to `n_fine` SGD steps (in-place parameter updates).
     *
     * @param batch_size Batch size; 0 means full-batch.
     * @param budget Global sample budget (0 disables budget stop).
     * @return Loss value from the last executed fine step.
     *
     * @details
     * Fine steps are standard SGD updates (`fine_opt_`) applied to the live network.
     * Each step:
     * - fetches a batch,
     * - creates a leaf `x` with gradients enabled,
     * - computes loss/gradients,
     * - applies the optimizer step.
     *
     * The loop early-exits if the global budget is reached.
     */
    double fine_solver(int batch_size, int budget) {
        tensor loss;
        tensor x;
        tensor u;

        for (int k = 0; k < n_fine; ++k) {
            tensor& batch_x_ref = this->data->train_next_batch(batch_size);
            x = batch_x_ref.clone().detach().set_requires_grad(true);

            u = this->net->forward(x);
            const std::vector<tensor> losses = this->data->losses(x, u, this->loss_fn);
            loss = torch::stack(losses).sum();
            
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
    /**
     * @brief Construct a ParaFlowS optimizer.
     *
     * @param data Dataset/geometry wrapper.
     * @param net Network to optimize.
     * @param lr_fine Fine learning rate.
     * @param n_fine Number of fine SGD steps per outer iteration.
     * @param loss_fn Scalar loss reducer (e.g. MSE).
     * @param n_coarse Maximum correction steps per outer iteration.
     */
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

    /**
     * @brief Train using the ParaFlowS scheme.
     *
     * @param batch_size Batch size; 0 means full batch.
     * @param budget Maximum number of samples to consume (0 disables budget stop).
     * @param max_iterations Maximum outer iterations.
     * @param verbose Enable progress logging and loss history tracking.
     * @return Result Training summary.
     */
    Result train(int batch_size, int budget, int max_iterations, bool verbose) override {
        if (verbose){
            std::cout << "Starting optimization with ParaFlowS..." << std::endl;
        }
        /** @details Training is fully in C++/torch; release the GIL to avoid blocking embedded Python. */
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
            /** @details Keep a local leaf copy with gradients enabled for PDE derivatives in test loss. */
            test = this->data->get_test().detach().clone().set_requires_grad(true);
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int it = 0; it < max_iterations; ++it) {
            /** @details 1) Coarse pass (compute `bff1` candidate parameters). */
            loss_coarse = coarse_solver(/*batch_size=*/0);

            if (budget > 0 && this->budget_used >= budget) {
                res.epoch = it + 1;
                break;
            }

            /** @details 2) Fine pass (updates live parameters). */
            loss_fine = fine_solver(batch_size, budget);

            /** @details 3) Correction loop (enforce monotonicity w.r.t. fine loss). */
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
                /** @details Apply candidate parameters: `params = bff1 + correction`. */
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

                if (loss_coarse <= loss_fine || j == 0) {
                    /** @details Accept: save current parameters as best-so-far. */
                    {
                        const auto params = this->net->parameters();
                        torch::NoGradGuard no_grad;
                        for (size_t i = 0; i < params.size(); ++i) {
                            params_old[i].copy_(params[i].detach());
                        }
                    }
                    loss_fine = loss_coarse;
                } else {
                    /** @details Reject: revert to best accepted parameters. */
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

            if (verbose) {
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

        /** @details Final loss on full training set. */
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