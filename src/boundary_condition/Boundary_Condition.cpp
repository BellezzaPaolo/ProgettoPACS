#include "Boundary_Condition.hpp"
#include <pybind11/numpy.h>
#include <algorithm>

/**
 * @file Boundary_Condition.cpp
 * @brief Implementation of generic boundary-point selection.
 *
 * The base `Boundary_Condition` provides:
 * - `filter(X)`: selects points on the boundary using the Python geometry mask
 *   (`geom.on_boundary`) and the user-provided predicate `on_boundary`.
 * - `collocation_points(X)`: currently equal to `filter(X)`.
 */

tensor Boundary_Condition::filter(const tensor& X) const {
    /** @details X is assumed to be of shape `[N, dim]`. */
    TORCH_CHECK(X.dim() == 2, "Boundary_Condition::filter expects X to be 2D [N, dim]");

    const auto N = X.size(0);
    const auto dim = X.size(1);

    /** @details Convert `X` to a NumPy array to call Python `geom.on_boundary` (DeepXDE-style API). */
    py::array_t<double> X_np({N, dim});
    auto buf = X_np.mutable_unchecked<2>();
    auto X_cpu = X.to(torch::kCPU).contiguous();

    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            buf(i, j) = X_cpu.index({i, j}).item<double>();
        }
    }

    /** @details `geom.on_boundary` returns a boolean array of shape `[N]`. */
    py::array_t<bool> index_bool_np = geom.attr("on_boundary")(X_np).cast<py::array_t<bool>>();
    auto idx_view = index_bool_np.unchecked<1>();

    std::vector<int64_t> indices;
    indices.reserve(N);
    for (int64_t i = 0; i < N; ++i) {
        /** @details Build a 1D tensor view for the i-th point, shape `[dim]`. */
        auto x_i = X_cpu.index({i});
        if (on_boundary(x_i, idx_view(i))) {
            indices.push_back(i);
        }
    }

    if (indices.empty()) {
        return torch::empty({0, dim}, X.options());
    }

    /** @details Stack the selected rows into the filtered tensor. */
    std::vector<tensor> rows;
    rows.reserve(indices.size());
    for (auto idx : indices) {
        rows.push_back(X_cpu.index({idx}).unsqueeze(0));
    }

    return torch::cat(rows, /*dim=*/0).to(X.device());
}

tensor Boundary_Condition::collocation_points(const tensor& X) const {
    return filter(X);
}
