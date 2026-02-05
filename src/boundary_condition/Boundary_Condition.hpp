#ifndef BOUNDARY_CONDITION_HPP
#define BOUNDARY_CONDITION_HPP

#include <iostream>
#include <functional>
#include <pybind11/embed.h>
#include <torch/torch.h>

/**
 * @file Boundary_Condition.hpp
 * @brief Abstract boundary-condition interface (DeepXDE-like) for PINNs.
 *
 * Boundary conditions are responsible for:
 * - selecting boundary collocation points from a candidate set
 * - computing an error tensor (residual) that will be converted to a scalar loss
 */

namespace py = pybind11;

using tensor = torch::Tensor;

/// \cond DOXYGEN_SHOULD_SKIP_THIS
// These pragmas silence GCC's -Wattributes visibility warnings. See the note at
// the end of this file for details.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif
/// \endcond

/**
 * @class Boundary_Condition
 * @brief Abstract base class for boundary conditions.
 * 
 * This is the parent class for all boundary conditions (Dirichlet, Neumann, Robin, Periodic, etc.).
 */
class Boundary_Condition {    
protected:
   /**
    * @brief Callback deciding whether a point lies on the boundary.
    *
    * @details
    * The callback signature is `on_boundary(x, on_bc_mask)` where:
    * - `x` is a 1D tensor view of one point, shape `[dim]`
    * - `on_bc_mask` is the boolean mask element returned by Python `geom.on_boundary(X)`.
    */
    std::function<bool(const tensor&, bool)> on_boundary;
    
   /** @brief Output component index this BC acts on. */
    int component;
    
   /** @brief Handle to the DeepXDE geometry object (Python). */
    py::handle geom;

public:
    /**
     * @brief Constructor for Boundary_Condition.
     * @param geom A geometry instance.
     * @param on_boundary A function that determines if a point is on the boundary.
     * @param component The output component satisfying this Boundary_Condition (default: 0).
     */
    Boundary_Condition(py::handle geom, std::function<bool(const tensor&, bool)> on_boundary, int component = 0):
        geom(geom), on_boundary(on_boundary), component(component) {};
 
    /**
     * @brief Virtual destructor.
     */
    virtual ~Boundary_Condition() = default;
    
    /**
     * @brief Filter points that are on the boundary.
     *
     * @details
     * This method calls the Python geometry predicate `geom.on_boundary(X)` to get
     * a boolean mask and then applies the user-provided `on_boundary(x_i, mask_i)`
     * callback to decide which points to keep.
     *
     * @param X Input points matrix `[N, dim]`.
     * @return Filtered points of shape `[M, dim]` with `M <= N`.
     */
    virtual tensor filter(const tensor& X) const;
    
    /**
     * @brief Get collocation points (points where Boundary_Condition is enforced).
     *
     * @details
     * Default implementation delegates to `filter(X)`.
     *
     * @param X Input points matrix `[N, dim]`.
     * @return Collocation points on the boundary `[M, dim]`.
     */
    virtual tensor collocation_points(const tensor& X) const;
    
    /**
     * @brief Compute the loss/error for this boundary condition.
     * Pure virtual method - must be implemented by derived classes.
     * 
     * @param X Full set of training points (DeepXDE-style: `train_x_all`).
     * @param inputs Batch input points `[N_batch, dim]`.
     * @param outputs Network outputs evaluated at `inputs`, shape `[N_batch, out_dim]`.
     * @param beg Beginning row (inclusive) of this BC segment inside the batch.
     * @param end Ending row (exclusive) of this BC segment inside the batch.
     * @return Error tensor of shape `[end-beg, 1]`.
     */
    virtual tensor error(const tensor& X, 
                          const tensor& inputs, 
                          const tensor& outputs, 
                          int beg, int end) const = 0;
};

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/**
 * @note Visibility warning (-Wattributes)
 * Some toolchains compile third-party headers (e.g. pybind11) with hidden symbol
 * visibility. Since `Boundary_Condition` stores pybind11 types (e.g. `py::handle`) and possibly
 * other types with different visibility, GCC may warn that `Boundary_Condition` has “greater
 * visibility” than the type of one of its fields.
 *
 * This matters mainly when exporting `Boundary_Condition` as part of a shared-library ABI.
 * In this project the executable is the main artifact, so we silence this
 * warning locally to keep build output clean.
 */

#endif