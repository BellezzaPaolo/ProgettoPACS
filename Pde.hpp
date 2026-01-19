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
    matrix train_x_bc;
    matrix test;

    std::vector<int> num_bcs;

    // TODO add the exact solution and its handle in the generation of the training and test set 

    matrix train_x;
    size_t previous_index;

public:
    // Constructor
    Pde(py::handle geom,
        std::function<std::vector<matrix>(const matrix&, const matrix&)> pde,
        std::vector<std::shared_ptr<Boundary_Condition>> bcs,
        int num_domain, int num_boundary, int num_test = 4, std::string train_distribution = "Hammersley");

    void generate_train_points();

    void generate_bc_points();

    void generate_test_points();

    matrix losses(
        const matrix& outputs,
        const matrix& inputs,
        const std::vector<std::function<double(const matrix&, const matrix&)>>& loss_fn
    );

    matrix& train_next_batch(int batch_size) const;
};
