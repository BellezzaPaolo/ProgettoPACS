#ifndef BOUNDARY_CONDITION_HPP
#define BOUNDARY_CONDITION_HPP

#include <iostream>
#include <functional>
#include <pybind11/embed.h>
#include <torch/torch.h>

namespace py = pybind11;

using tensor = torch::Tensor;

/**
 * @class Boundary_Condition
 * @brief Abstract base class for boundary conditions.
 * 
 * This is the parent class for all boundary conditions (Dirichlet, Neumann, Robin, Periodic, etc.).
 */
class __attribute__((visibility("hidden"))) Boundary_Condition {    
protected:
    // Function to check if points are on the boundary
    // x is a 1D tensor view of a point with shape [dim]
    std::function<bool(const tensor&, bool)> on_boundary;
    
    // Output component that this Boundary_Condition is applied to
    int component;
    
    // Reference to deepXDE geometry class
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
     * @param X Input points matrix (N x dim).
     * @return Filtered points that satisfy the boundary condition.
     */
    virtual tensor filter(const tensor& X) const;
    
    /**
     * @brief Get collocation points (points where Boundary_Condition is enforced).
     * @param X Input points matrix.
     * @return Collocation points on the boundary.
     */
    virtual tensor collocation_points(const tensor& X) const;
    
    /**
     * @brief Compute the loss/error for this boundary condition.
     * Pure virtual method - must be implemented by derived classes.
     * 
     * @param X Points on the boundary.
     * @param inputs Network inputs.
     * @param outputs Network outputs.
     * @param beg Beginning index.
     * @param end Ending index.
     * @param aux_var Auxiliary variables (optional, default: nullptr).
     * @return Error vector of size (end-beg) x 1.
     */
    virtual tensor error(const tensor& X, 
                          const tensor& inputs, 
                          const tensor& outputs, 
                          int beg, int end) const = 0;
};

#endif