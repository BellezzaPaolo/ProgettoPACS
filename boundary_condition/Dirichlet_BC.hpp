#ifndef DIRICHLET_BC_HPP
#define DIRICHLET_BC_HPP

#include "Boundary_Condition.hpp"
#include <pybind11/embed.h>


/**
 * @class Dirichlet_BC
 * @brief Dirichlet boundary condition: y(x) = func(x).
 */
class __attribute__((visibility("hidden"))) Dirichlet_BC : public Boundary_Condition {
private:
    std::function<tensor(const tensor&)> func;
    
public:
    /**
     * @brief Constructor for Dirichlet_BC.
     * @param geom Geometry instance.
     * @param func Function that specifies the boundary value.
     * @param on_boundary Boundary check function.
     * @param component Output component (default: 0).
     */
    Dirichlet_BC(py::handle geom, std::function<tensor(const tensor&)> func, std::function<bool(const tensor&, bool)> on_boundary, int component = 0): 
        Boundary_Condition(geom, on_boundary, component), func(func) {};
    
    /**
     * @brief Virtual destructor.
     */
    virtual ~Dirichlet_BC();
         

    /**
     * @brief Compute the Dirichlet BC error.
     */
    tensor error(const tensor& X, 
              const tensor& inputs, 
              const tensor& outputs, 
              int beg, int end) const override;
};

#endif