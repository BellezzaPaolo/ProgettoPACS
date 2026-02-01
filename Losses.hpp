#ifndef LOSSES_HPP
#define LOSSES_HPP

#include <functional>
#include <string>
#include <unordered_map>
#include <stdexcept>

#include <torch/torch.h>

using tensor = torch::Tensor;

namespace losses {
using ResidualLossFn = std::function<tensor(const tensor& r)>;

inline ResidualLossFn get_loss(const std::string& identifier) {
    // It's important to notice that by construction the values 
    // that arrive here are already the errors and this is the reason why I don't use the built-in functions of torch
    // that would require another tensor intitialized to 0 as DeepXDE does

    if (identifier == "mean squared error" || identifier == "MSE" || identifier == "mse") {
        // testing, it gets that takes less in mean but with higher variance than:
        // return torch::mse_loss(r, torch::zeros_like(r))
        return [](const tensor& r) { return r.square().mean(); };
    }
    if (identifier == "mean absolute error" || identifier == "MAE" || identifier == "mae") {
        return [](const tensor& r) { return r.abs().mean(); };
    }

    throw std::runtime_error(
        "losses::get_residual: loss '" + identifier + "' needs y_true, not just residual r");
}


} // namespace losses

#endif