#ifndef DIRICHLET_BC_HPP
#define DIRICHLET_BC_HPP

#include "Boundary_Condition.hpp"
#include <pybind11/embed.h>


/**
 * @class DirichletBC
 * @brief Dirichlet boundary condition: y(x) = func(x).
 */
class __attribute__((visibility("hidden"))) DirichletBC : public Boundary_Condition {
private:
    std::function<matrix(const matrix&)> func;
    
public:
    /**
     * @brief Constructor for DirichletBC.
     * @param geom Geometry instance.
     * @param func Function that specifies the boundary value.
     * @param on_boundary Boundary check function.
     * @param component Output component (default: 0).
     */
    DirichletBC(py::handle geom, std::function<matrix(const matrix&)> func, std::function<bool(const vector&, bool)> on_boundary, int component = 0): 
        Boundary_Condition(geom, on_boundary, component), func(func) {};
    
         

    /**
     * @brief Compute the Dirichlet BC error.
     */
    matrix error(const matrix& X, 
                  const matrix& inputs, 
                  const matrix& outputs, 
                  int beg, int end) const override;
};

#endif