#ifndef LOSSES_HPP
#define LOSSES_HPP

/**
 * @file Losses.hpp
 * @brief Small registry of residual-based loss functions.
 *
 * This header provides a single factory function, losses::get_loss(), which
 * returns a callable computing a scalar loss from a residual tensor.
 *
 * @details
 * In this codebase, the input passed to the loss is assumed to already be the
 * residual (i.e. an error signal). Therefore the returned functors operate on
 * the residual directly and do not require a separate `y_true` tensor.
 */

#include <functional>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <torch/torch.h>

using tensor = torch::Tensor;

namespace losses {
/**
 * @brief Callable type for residual-based loss functions.
 *
 * The callable takes a residual tensor @p r and returns a scalar tensor loss.
 */
using ResidualLossFn = std::function<tensor(const tensor& r)>;

/**
 * @brief Get a residual-based loss functor by identifier.
 *
 * @param identifier Human-readable or short identifier of the loss.
 *
 * @return A function `f(r)` that maps a residual tensor @p r to a scalar loss.
 *
 * @throws std::runtime_error If the identifier is not recognized.
 *
 * @par Supported identifiers
 * - Mean squared error: `"mean squared error"`, `"MSE"`, `"mse"`
 * - Mean absolute error: `"mean absolute error"`, `"MAE"`, `"mae"`
 *
 * @note
 * The input to the returned functor is assumed to be the residual itself.
 * This avoids allocating an extra zero tensor as would be required by APIs
 * like `torch::mse_loss(y_pred, y_true)`.
 */
inline ResidualLossFn get_loss(const std::string& identifier) {
    /**
     * @details
     * In this codebase the values passed here are already residuals/errors.
     * We therefore implement the loss directly on the residual to avoid
     * allocating an explicit zero target tensor (as APIs like `torch::mse_loss`
     * would require).
     */

    if (identifier == "mean squared error" || identifier == "MSE" || identifier == "mse") {
        /**
         * @note
         * This is equivalent to `torch::mse_loss(r, zeros_like(r))` but avoids the
         * extra allocation for the zero tensor.
         */
        return [](const tensor& r) { return r.square().mean(); };
    }
    if (identifier == "mean absolute error" || identifier == "MAE" || identifier == "mae") {
        return [](const tensor& r) { return r.abs().mean(); };
    }

    throw std::runtime_error(
        "losses::get_loss: unknown loss identifier '" + identifier + "'");
}


} // namespace losses

#endif