#include "Boundary_Condition.hpp"
#include <algorithm>

matrix Boundary_Condition::filter(const matrix& X) const {
    std::vector<int> indices;
    bool flag;

    // Check each point to see if it's on the boundary
    for (int i = 0; i < X.rows(); ++i) {
        vector point = X.row(i);

        flag = geom.attr("on_boundary")(point).cast<bool>();
        if (on_boundary(point, flag)){
            indices.push_back(i);
        }
    }
    
    // Build filtered matrix
    matrix filtered(indices.size(), X.cols());
    for (size_t i = 0; i < indices.size(); ++i) {
        filtered.row(i) = X.row(indices[i]);
    }
    
    return filtered;
}

matrix Boundary_Condition::collocation_points(const matrix& X) const {
    return filter(X);
}
