#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <functional>
#include <pybind11/embed.h>
#include "boundary_condition/Boundary_Condition.hpp"

namespace py = pybind11;

using matrix = Eigen::MatrixXd;
using vector = Eigen::VectorXd;

class __attribute__((visibility("hidden"))) Pde {
private:
    // Geometry interface (Python: geom)
    py::handle geom;
    int phisical_dim;

    // PDE residual function
    std::function<std::vector<matrix>(const matrix& inputs, const matrix& outputs)> pde;

    // Boundary conditions
    std::vector<std::shared_ptr<Boundary_Condition>> bcs;

    int num_domain;
    int num_boundary;
    int num_test;

    std::string train_distribution;

    matrix train_x_pde; // is train_x_all of deepXDE, following the TODO comment has been changed the name
    std::vector<matrix> train_x_bc;
    matrix test;

    std::vector<int> num_bcs;

    // TODO add the exact solution and its handle in the generation of the training and test set 

    matrix train_x;
    bool is_train_x_initialized;

    // Batching state
    int idx_pde_prec;
    std::vector<int> idx_bc_prec;
    unsigned int epoch_seed = 123;
    std::vector<int> num_bcs_batch; // sizes for current batch BC splits
    int training_batch_size;

    // this functions are made private because the generation of points in the domain
    // can also take points on the boundary so it's not sure that train_x_bcs is exactly of 
    // length num_boundary. So by now are accessible only through the constructor.
    // TODO: make a function to regenerate teh point and consider resize test and train_x_bc.
    void generate_train_points();

    void generate_bc_points();

    void generate_test_points();
public:
    // Constructor
    Pde(py::handle geom,
        std::function<std::vector<matrix>(const matrix&, const matrix&)> pde,
        std::vector<std::shared_ptr<Boundary_Condition>> bcs,
        int Num_domain, int Num_boundary, int Num_test = 4, std::string train_distribution = "Hammersley");

    std::vector<double> losses(
        const matrix& inputs,
        const matrix& outputs,
        const std::function<double(const matrix&)>& loss_fn
    );

    // Build next training batch; if batch_size <= 0, use full set
    matrix& train_next_batch(const int batch_size = 0);
};
