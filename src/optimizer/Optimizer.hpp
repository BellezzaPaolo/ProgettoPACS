#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

/**
 * @file Optimizer.hpp
 * @brief Base optimizer interface and training result container.
 *
 * The project follows a DeepXDE-like design:
 * - `Model` owns the dataset (`Pde`) and the network.
 * - Concrete optimizers implement `Optimizer::train(...)`.
 * - Training metadata is returned via `Result`.
 */

#include <functional>
#include <memory>
#include <vector>

#include <pybind11/embed.h>
#include <torch/torch.h>

#include "../Pde.hpp"

namespace py = pybind11;

using tensor = torch::Tensor;

/**
 * @brief Training summary returned by optimizers.
 *
 * @details
 * This struct is intentionally simple and serializable (e.g. to CSV).
 * Many fields are filled by the optimizer implementation.
 */
struct Result {
    /// Number of training samples consumed so far (budget accounting).
    int budget_used = 0;
    /// Final scalar loss value computed at the end of training.
    double final_loss = 0.0;
    /// Last completed iteration/epoch counter (optimizer-dependent semantics).
    int epoch = 0;
    /// Total training wall time in milliseconds.
    double total_time_ms = 0.0;
    /// Requested budget passed to `train` (in number of samples).
    int budget = 0;
    /// Batch size used during training (0 means full batch in this project).
    int batch_size = 0;
    /// ParaFlowS-only: number of fine steps per outer iteration; 0 if not used.
    int n_fine = 0; /**< If the optimizer doesn't use it, it remains 0. */
    /// Learning rate used by the optimizer.
    double lr = 0;
};

template <class NetT>
class Optimizer {
protected:
    std::shared_ptr<Pde> data;
    std::shared_ptr<NetT> net;
    std::function<tensor(const tensor&)> loss_fn;

    /**
     * @brief Loss histories for plotting/diagnostics.
     * @details Derived optimizers can `push_back` into these vectors when they log.
     */
    std::vector<double> loss_history_train;
    std::vector<double> loss_history_test;

    bool use_test;

    int budget_used = 0;

public:
    /**
     * @brief Construct an optimizer bound to a dataset and a network.
     * @param data Dataset/geometry wrapper.
     * @param net Neural network to be trained.
     * @param loss_fn Scalar loss reducer used by the PDE and BC terms.
     */
    Optimizer(std::shared_ptr<Pde> data,
              std::shared_ptr<NetT> net,
              std::function<tensor(const tensor&)> loss_fn)
        : data(std::move(data)), net(std::move(net)), loss_fn(loss_fn) {
            use_test = this->data->has_test_set();
        }

    virtual ~Optimizer() = default;

    /**
     * @brief Return the training loss history.
     *
     * @details
     * This vector is populated by concrete optimizers when `verbose` is enabled.
     * It is safe to be empty if the optimizer never logged.
     */
    const std::vector<double>& get_loss_history_train() const { return loss_history_train; }

    /**
     * @brief Return the test loss history.
     *
     * @details
     * This vector is populated only if a test set is available (`Pde::has_test_set()`).
     * It is safe to be empty.
     */
    const std::vector<double>& get_loss_history_test() const { return loss_history_test; }

    /**
     * @brief Plot the stored cost history using Python/matplotlib.
     *
     * @details
     * This is a convenience helper for quick debugging/visualization.
     * Since matplotlib is a Python library, this method acquires the Python GIL.
     *
     * - Train history is plotted with `loglog`.
     * - If test history exists, it is overlaid.
     */
    void plot_cost_history() const {
        py::gil_scoped_acquire gil;
        py::module_ plt = py::module_::import("matplotlib.pyplot");

        py::list py_train(loss_history_train.size());
        for (size_t i = 0; i < loss_history_train.size(); ++i) {
            py_train[i] = loss_history_train[i];
            py::print(py_train[i]);
        }
        plt.attr("loglog")(py_train, py::arg("label") = "train");

        if (!loss_history_test.empty()) {
            py::list py_test(loss_history_test.size());
            for (size_t i = 0; i < loss_history_test.size(); ++i) {
                py_test[i] = loss_history_test[i];
            }
            plt.attr("plot")(py_test, py::arg("label") = "test");
            plt.attr("legend")();
        }

        plt.attr("xlabel")("Iteration");
        plt.attr("ylabel")("Cost");
        plt.attr("title")("Cost History");
        plt.attr("show")();
    }

    /**
     * @brief Train the network.
     *
     * @param batch_size Batch size; by convention, 0 means full batch.
     * @param budget Maximum number of samples to consume (0 disables budget stop).
     * @param max_iterations Maximum optimizer iterations.
     * @param verbose Enable progress logging.
     * @return Result Training summary.
     */
    virtual Result train(int batch_size, int budget, int max_iterations, bool verbose) = 0;

};

#endif