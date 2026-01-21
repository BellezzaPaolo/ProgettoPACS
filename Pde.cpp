#include <iostream>
#include <stdexcept>
#include <pybind11/eigen.h>
#include "Pde.hpp"

Pde::Pde(py::handle geom,
    std::function<std::vector<matrix>(const matrix&, const matrix&)> pde,
    std::vector<std::shared_ptr<Boundary_Condition>> bcs,
    int num_domain, int num_boundary, int num_test, std::string train_distribution):
    geom(geom), pde(pde), bcs(bcs), num_domain(num_domain), num_boundary(num_boundary), train_distribution(train_distribution), num_test(num_test){
        if(num_domain <= 0){
            throw std::runtime_error("num_domain must be a positive integer");
        }
        if(num_boundary <= 0){
            throw std::runtime_error("num_boundary must be a positive integer");
        }

        phisical_dim = geom.attr("dim").cast<int>();
        // initialize structures to store training set
        train_x_pde = matrix::Zero(num_domain + num_boundary, phisical_dim);
        train_x_bc = matrix::Zero(num_boundary, phisical_dim);
        num_bcs.resize(bcs.size());

        if(num_test > 0){
            test = matrix::Zero(num_test + num_boundary, phisical_dim);
            generate_test_points();
        }
    };

void Pde::generate_train_points(){
    py::object X_py;
    py::object X_bc_py;

    if(train_distribution == "uniform"){
        X_py = geom.attr("uniform_points")(num_domain, py::arg("boundary")=false);
        X_bc_py = geom.attr("uniform_boundary_points")(num_boundary);
    }
    else{
        X_py = geom.attr("random_points")(num_domain, train_distribution);
        X_bc_py = geom.attr("random_boundary_points")(num_boundary, train_distribution);
    }

    // X.block(start_row, start_col, num_rows, num_cols)
    train_x_pde.block(0,0,num_domain, phisical_dim) = X_py.cast<matrix>();
    train_x_pde.block(num_domain,0,num_boundary, phisical_dim) = X_bc_py.cast<matrix>();

    return;
}

void Pde::generate_bc_points(){
    int i = 0;
    int index = 0;
    for( auto bc: bcs){
        matrix X_bc = bc->collocation_points(train_x_pde);

        num_bcs[i] = X_bc.rows();

        train_x_bc.block(index, 0, num_bcs[i], phisical_dim)= X_bc;

        index += num_bcs[i];
        i++;
    }
    return;
}

void Pde::generate_test_points(){
    if(num_test > 0){
        // the distribution here is random beacuse for same geometries the "uniform_point" function is not implemented
        py::object X_py = geom.attr("random_points")(num_test);
        test.block(0, 0, num_test, phisical_dim) = X_py.cast<matrix>();
        test.block(num_test, 0, num_boundary, phisical_dim) = train_x_bc;
    }
    else{
        throw std::runtime_error("num_test is Zero so no test set");
    }
    return;
}

matrix Pde::losses(
    const matrix& outputs,
    const matrix& inputs,
    const std::vector<std::function<double(const matrix&, const matrix&)>>& loss_fn
){
    matrix res;
    // come si danno gli input a pde? (2 o 3) chat


    return res;
}

matrix& Pde::train_next_batch(int batch_size) const{
    matrix res;
    
    return res;
}