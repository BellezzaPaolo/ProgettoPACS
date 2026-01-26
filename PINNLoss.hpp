#ifndef PINNLOSS_HPP
#define PINNLOSS_HPP

#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include "FNN.hpp"

using namespace autodiff;

// ============================================================================
// Spatial Derivative Utilities (wrapping autodiff library functions)
// ============================================================================

// Compute Laplacian (∇²u = d²u/dx² + d²u/dy² + ...) using autodiff
// Simplified: computes trace of Hessian (sum of diagonal elements)
template <activation_type A>
double compute_laplacian(
    FNN<A, double>& net,
    const vector_t<double>& x
) {
    double laplacian = 0.0;
    int dim = x.size();
    
    // For each dimension i, compute d²u/dx_i²
    for (int i = 0; i < dim; ++i) {
        // Use dual2nd to get second derivative w.r.t. x_i
        vector_t<dual2nd> x_dual = x.template cast<dual2nd>();
        x_dual(i).val().val() = x(i);
        x_dual(i).val().eps() = 1.0;  // marks x_i for first deriv
        x_dual(i).eps().val() = 1.0;  // marks x_i for second deriv
        
        dual2nd u = net.forward(x_dual)(0);
        laplacian += u.eps().eps();  // d²u/dx_i²
    }
    
    return laplacian;
}

#endif