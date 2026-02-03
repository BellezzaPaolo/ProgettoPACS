#pragma once

#include <torch/torch.h>

/**
 * @file Differential_Operators.hpp
 * @brief Differential operators implemented with LibTorch autograd.
 *
 * This header provides small utilities (e.g. Laplacian) used by PINN PDE
 * residual definitions. Functions are header-only for convenience.
 */

/**
 * @namespace differential_operators
 * @brief Autograd-based differential operators.
 */
namespace differential_operators {

/// Alias used across the project for LibTorch tensors.
using tensor = torch::Tensor;

/**
 * @brief Compute the Laplacian of a scalar field $u(x)$ using torch autograd.
 *
 * @details
 * For each sample $x \in \mathbb{R}^{\mathrm{dim}}$, this returns
 * $$\Delta u(x) = \sum_{i=1}^{\mathrm{dim}} \frac{\partial^2 u}{\partial x_i^2}.$$
 *
 * Expected tensor shapes:
 * - `x`: `[N, dim]` with `requires_grad=true`
 * - `u`: `[N, 1]` or `[N]` (or `[N, out_dim]` with a selected `component`)
 *
 * @param u Field values evaluated at `x`.
 * @param x Input coordinates. Must have `requires_grad=true`.
 * @param component Component index to use when `u` is `[N, out_dim]`.
 * @return Tensor of shape `[N, 1]` containing the Laplacian values.
 */
inline tensor laplacian(const tensor& u, const tensor& x, int64_t component = 0) {
    TORCH_CHECK(x.defined(), "laplacian: x is undefined");
    TORCH_CHECK(u.defined(), "laplacian: u is undefined");
    TORCH_CHECK(x.dim() == 2, "laplacian: x must be [N, dim]");
    TORCH_CHECK(x.requires_grad(), "laplacian: x must have requires_grad=true");

    using torch::indexing::Slice;

    tensor u_comp;
    if (u.dim() == 2) {
        TORCH_CHECK(u.size(0) == x.size(0), "laplacian: batch mismatch between u and x");
        TORCH_CHECK(component >= 0 && component < u.size(1), "laplacian: component out of range");
        u_comp = u.index({Slice(), component});
    } else if (u.dim() == 1) {
        TORCH_CHECK(u.size(0) == x.size(0), "laplacian: batch mismatch between u and x");
        u_comp = u;
    } else {
        TORCH_CHECK(false, "laplacian: u must be [N], [N,1], or [N,out_dim]");
    }

    /** @details First derivatives: `grad_u = du/dx`, shape `[N, dim]`. */
    tensor grad_u = torch::autograd::grad(
        /*outputs=*/{u_comp.sum()},
        /*inputs=*/{x},
        /*grad_outputs=*/{},
        /*retain_graph=*/true,
        /*create_graph=*/true
    )[0];

    const int64_t dim = x.size(1);
    tensor lap = torch::zeros({x.size(0)}, x.options());

    /** @details Second derivatives: $\sum_i \partial^2 u/\partial x_i^2$ (trace of Hessian). */
    for (int64_t d = 0; d < dim; ++d) {
        tensor du_d = grad_u.index({Slice(), d});

        tensor grad2 = torch::autograd::grad(
            /*outputs=*/{du_d.sum()},
            /*inputs=*/{x},
            /*grad_outputs=*/{},
            /*retain_graph=*/true,
            /*create_graph=*/true
        )[0];

        lap = lap + grad2.index({Slice(), d});
    }

    return lap.unsqueeze(1);
}

} // namespace differential_operators
