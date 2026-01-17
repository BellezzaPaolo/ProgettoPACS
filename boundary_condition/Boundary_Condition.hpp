#ifndef BOUNDARY_CONDITION_HPP
#define BOUNDARY_CONDITION_HPP

#include <iostream>
#include <functional>
#include <pybind11/embed.h>
#include <Eigen/Dense>

namespace py = pybind11;

using matrix = Eigen::MatrixXd;
using vector = Eigen::VectorXd;

/**
 * @class Boundary_Condition
 * @brief Abstract base class for boundary conditions.
 * 
 * This is the parent class for all boundary conditions (Dirichlet, Neumann, Robin, Periodic, etc.).
 */
class __attribute__((visibility("hidden"))) Boundary_Condition {    
protected:
    // Function to check if points are on the boundary
    std::function<bool(const vector&, bool)> on_boundary;
    
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
    Boundary_Condition(py::handle geom, std::function<bool(const vector&, bool)> on_boundary, int component = 0):
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
    virtual matrix filter(const matrix& X) const;
    
    /**
     * @brief Get collocation points (points where Boundary_Condition is enforced).
     * @param X Input points matrix.
     * @return Collocation points on the boundary.
     */
    virtual matrix collocation_points(const matrix& X) const;
    
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
    virtual matrix error(const matrix& X, 
                          const matrix& inputs, 
                          const matrix& outputs, 
                          int beg, int end) const = 0;
};

#endif