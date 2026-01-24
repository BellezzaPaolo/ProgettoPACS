#include "Dirichlet_BC.hpp"

#include <stdexcept>
#include <algorithm>

Dirichlet_BC::~Dirichlet_BC() = default;

matrix Dirichlet_BC::error(const matrix& X, 
                            const matrix& inputs, 
                            const matrix& outputs, 
                            int beg, int end) const {

    // Extract the relevant portion of X
    // X.block(start_row, start_col, num_rows, num_cols)
    // matrix X_slice = X.block(beg, 0, end - beg, X.cols());
    
    // Compute boundary values using the function
    matrix values = func(X.block(beg, 0, end - beg, X.cols()));
    
    // Check dimensions
    if (values.cols() != 1) {
        throw std::runtime_error("Dirichlet_BC function should return an array of shape N by 1 for each "
            "component. Use argument 'component' for different output components.");
    }
    
    // Compute error: outputs - values
    matrix error_matrix = outputs.block(beg, component, end - beg, 1) - values;
    
    return error_matrix;
}

