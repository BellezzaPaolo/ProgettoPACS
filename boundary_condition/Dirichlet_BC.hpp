#ifndef DIRICHLET_BC_HPP
#define DIRICHLET_BC_HPP

#include "Boundary_Condition.hpp"
#include <pybind11/embed.h>

/**
 * @file Dirichlet_BC.hpp
 * @brief Dirichlet boundary condition implementation.
 */

/**
 * @class Dirichlet_BC
 * @brief Dirichlet boundary condition: y(x) = func(x).
 *
 * @details
 * The error returned is:
 * \f[ e(x) = u(x) - g(x) \f]
 * where `u(x)` is the network output and `g(x)` is the provided boundary target.
 */
class __attribute__((visibility("hidden"))) Dirichlet_BC : public Boundary_Condition {
private:
    std::function<tensor(const tensor&)> func;
    
public:
    /**
     * @brief Constructor for Dirichlet_BC.
     * @param geom DeepXDE geometry handle (Python).
     * @param func Function that returns boundary target values `g(x)` as a tensor of shape `[N, 1]`.
     * @param on_boundary Boundary check callback used in `filter()`.
     * @param component Output component (default: 0).
     */
    Dirichlet_BC(py::handle geom, std::function<tensor(const tensor&)> func, std::function<bool(const tensor&, bool)> on_boundary, int component = 0): 
        Boundary_Condition(geom, on_boundary, component), func(func) {};
    
    /**
     * @brief Virtual destructor.
     */
    virtual ~Dirichlet_BC();
         

    /**
        * @brief Compute the Dirichlet BC error tensor on a batch segment.
        *
        * @param X Full training set (unused here but kept for interface compatibility).
        * @param inputs Batch input points `[N_batch, dim]`.
        * @param outputs Network outputs evaluated at `inputs`, shape `[N_batch, out_dim]`.
        * @param beg Beginning row (inclusive) of this BC segment inside the batch.
        * @param end Ending row (exclusive) of this BC segment inside the batch.
        * @return Error tensor of shape `[end-beg, 1]`.
     */
    tensor error(const tensor& X, 
              const tensor& inputs, 
              const tensor& outputs, 
              int beg, int end) const override;
};

#endif