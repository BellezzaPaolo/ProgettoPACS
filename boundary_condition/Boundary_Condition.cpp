#include "Boundary_Condition.hpp"
#include <pybind11/eigen.h>
#include <algorithm>

matrix Boundary_Condition::filter(const matrix& X) const {
    Eigen::Index j = 0;
    std::vector<size_t> index;

    // Check each point to see if it's on the boundary
    Eigen::Array<bool, Eigen::Dynamic, 1> index_bool =  geom.attr("on_boundary")(X).cast<Eigen::Array<bool, Eigen::Dynamic, 1>>();

    for(size_t i = 0; i< X.rows(); ++i){
        if(on_boundary(X.row(i), index_bool[i])){
            index.push_back(i);
            j++;
        }
    }

    matrix filtered(j, X.cols());
    // Build filtered matrix
    for(size_t i = 0; i < j; ++i){
        filtered.row(i) = X.row(index[i]);
    }

    return filtered;
}

matrix Boundary_Condition::collocation_points(const matrix& X) const {
    return filter(X);
}
