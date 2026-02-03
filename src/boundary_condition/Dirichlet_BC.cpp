#include "Dirichlet_BC.hpp"

#include <stdexcept>
#include <algorithm>

/**
 * @file Dirichlet_BC.cpp
 * @brief Implementation of the Dirichlet boundary condition.
 */

Dirichlet_BC::~Dirichlet_BC() = default;

tensor Dirichlet_BC::error(const tensor& X, 
                            const tensor& inputs, 
                            const tensor& outputs, 
                            int beg, int end) const {

    TORCH_CHECK(X.dim() == 2, "Dirichlet_BC::error expects X to be 2D [N, dim]");
    TORCH_CHECK(outputs.dim() == 2, "Dirichlet_BC::error expects outputs to be 2D [N, n_out]");

    const auto N = X.size(0);
    const auto dim = X.size(1);

    TORCH_CHECK(beg >= 0 && end <= N && beg <= end, "Dirichlet_BC::error: invalid beg/end range");

    const int64_t num_rows = end - beg;
    if (num_rows == 0) {
        return torch::zeros({0, 1}, outputs.options());
    }

    /** @details Slice `X` for the relevant rows. */
    auto X_slice = X.index({torch::indexing::Slice(beg, end), torch::indexing::Slice()});

    /** @details Compute boundary target values, shape `[num_rows, 1]`. */
    tensor values = func(X_slice);
    if (values.dim() != 2 || values.size(1) != 1 || values.size(0) != num_rows) {
        throw std::runtime_error("Dirichlet_BC function should return a tensor of shape (N, 1) for each "
            "component. Use argument 'component' for different output components.");
    }

    /** @details Slice outputs for the specified component, shape `[num_rows, 1]`. */
    auto out_slice = outputs.index({torch::indexing::Slice(beg, end), component});
    out_slice = out_slice.view({num_rows, 1});

    /** @details Error tensor: `u(x) - g(x)` (outputs minus target values). */
    return out_slice - values;
}

