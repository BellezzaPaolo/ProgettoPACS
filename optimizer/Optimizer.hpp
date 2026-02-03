#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <functional>
#include <memory>
#include <vector>

#include <pybind11/embed.h>
#include <torch/torch.h>

#include "../Pde.hpp"

namespace py = pybind11;

using tensor = torch::Tensor;

struct Result {
    int budget_used = 0;
    double final_loss = 0.0;
    int epoch = 0;
    double total_time_ms = 0.0;
    int budget = 0;
    int batch_size = 0;
    int n_fine = 0; // if the optimizer doesn't use it, this 0 will be the sign
    double lr = 0;
};

template <class NetT>
class Optimizer {
protected:
    std::shared_ptr<Pde> data;
    std::shared_ptr<NetT> net;
    std::function<tensor(const tensor&)> loss_fn;

    // Keep empty; derived optimizers can push_back each time they log.
    std::vector<double> loss_history_train;
    std::vector<double> loss_history_test;

    bool use_test;

    int budget_used = 0;

public:
    Optimizer(std::shared_ptr<Pde> data,
              std::shared_ptr<NetT> net,
              std::function<tensor(const tensor&)> loss_fn)
        : data(std::move(data)), net(std::move(net)), loss_fn(loss_fn) {
            use_test = this->data->has_test_set();
        }

    virtual ~Optimizer() = default;

    const std::vector<double>& get_loss_history_train() const { return loss_history_train; }
    const std::vector<double>& get_loss_history_test() const { return loss_history_test; }

    void plot_cost_history() const {
        py::gil_scoped_acquire gil;
        py::module_ plt = py::module_::import("matplotlib.pyplot");

        py::list py_train(loss_history_train.size());
        for (size_t i = 0; i < loss_history_train.size(); ++i) {
            py_train[i] = loss_history_train[i];
        }
        plt.attr("plot")(py_train, py::arg("label") = "train");

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

    virtual Result train(int batch_size, int budget, int max_iterations, bool verbose) = 0;

};

#endif