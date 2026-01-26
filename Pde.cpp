#include <iostream>
#include <stdexcept>
#include <numeric>
#include <random>
#include <pybind11/eigen.h>
#include "Pde.hpp"

Pde::Pde(py::handle geom,
    std::function<std::vector<matrix>(const matrix&, const matrix&)> pde,
    std::vector<std::shared_ptr<Boundary_Condition>> bcs,
    int Num_domain, int Num_boundary, int Num_test, std::string train_distribution):
    geom(geom), pde(pde), bcs(bcs), num_domain(Num_domain), num_boundary(Num_boundary), num_test(Num_test), train_distribution(train_distribution){
        if(num_domain <= 0){
            throw std::runtime_error("num_domain must be a positive integer");
        }
        if(Num_boundary <= 0){
            throw std::runtime_error("num_boundary must be a positive integer");
        }

        phisical_dim = geom.attr("dim").cast<int>();
        // initialize structures to store training set
        train_x_pde = matrix::Zero(num_domain + num_boundary, phisical_dim);
        train_x_bc.resize(bcs.size());// = matrix::Zero(num_domain + num_boundary, phisical_dim);
        num_bcs.resize(bcs.size());
        
        // Generate training and BC points once
        generate_train_points();
        generate_bc_points();
        is_train_x_initialized = false;

        if(num_test > 0){
            test = matrix::Zero(num_test + num_boundary, phisical_dim);
            generate_test_points();
        }

        // allocate the maximum space needed for the batch of the current iteration
        train_x = matrix::Zero(num_domain + num_boundary + num_boundary, phisical_dim);

        idx_pde_prec = 0;
        idx_bc_prec.resize(bcs.size());
        epoch_seed = 123;
        num_bcs_batch.resize(bcs.size()); // sizes for current batch BC splits
        training_batch_size = 0;
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
    int real_points_on_boundary = 0;
    for(size_t i = 0; i < bcs.size(); ++i){
        train_x_bc[i] = bcs[i]->collocation_points(train_x_pde);

        num_bcs[i] = train_x_bc[i].rows();

        real_points_on_boundary += num_bcs[i];

        // std::cout << " bc "<< i << " num_bcs: " << num_bcs[i] << " train_x_bc[i] " << train_x_bc[i] <<std::endl;
    }
    // reassign num_boundary
    num_boundary = real_points_on_boundary;
    // std::cout << " num_boundary " << num_boundary << std::endl;
    return;
}

void Pde::generate_test_points(){
    if(num_test > 0){
        // the distribution here is random beacuse for same geometries the "uniform_point" function is not implemented
        py::object X_py = geom.attr("random_points")(num_test);
        test.block(0, 0, num_test, phisical_dim) = X_py.cast<matrix>();
        for(matrix elem_bc : train_x_bc){
            test.block(num_test, 0, elem_bc.rows(), phisical_dim) = elem_bc;
        }
    }
    else{
        throw std::runtime_error("num_test is zero so no test set");
    }
    return;
}

std::vector<double> Pde::losses(
    const matrix& inputs,
    const matrix& outputs,
    const std::function<double(const matrix&)>& loss_fn
){
    // Validate shapes
    if (inputs.rows() != outputs.rows()) {
        std::cout << "inputs rows are "<< inputs.rows() << "and outputs rows are "<< outputs.rows()<< std::endl; 
        throw std::runtime_error("inputs and outputs must have the same number of rows");
    }

    // Choose BC counts: use batch-specific if available and matching this inputs size
    std::vector<int> nb;
    if (inputs.rows() == training_batch_size) {
        nb = num_bcs_batch;
    } else {
        nb = num_bcs;
    }

    // Compute cumulative BC indices: bcs_start[i] is the starting row for BC i
    // The batch structure is: [BC0_points | BC1_points | ... | BCn_points | PDE_points]
    // bcs_start = [0, num_bcs[0], num_bcs[0]+num_bcs[1], ..., total_bc]

    std::vector<int> bcs_start;
    bcs_start.push_back(0);
    for (size_t i = 0; i < nb.size(); ++i) {
        bcs_start.push_back(bcs_start.back() + nb[i]);
    }
    
    const int total_bc = bcs_start.back();
    if (total_bc > inputs.rows()) {
        throw std::runtime_error("num_bcs exceeds provided inputs/outputs rows");
    }

    // Collect all loss values (PDE losses first, then BC losses)
    std::vector<double> losses_val;

    // PDE residuals on the remaining rows (after all BC points)
    const int pde_rows = inputs.rows() - total_bc;
    std::vector<matrix> residuals;
    
    if (pde_rows > 0) {
        // User-defined residual(s): vector of matrices
        residuals = pde(inputs.block(total_bc, 0, pde_rows, inputs.cols()), 
                       outputs.block(total_bc, 0, pde_rows, outputs.cols()));
        
        // Compute PDE losses using loss_fn
        for (size_t i = 0; i < residuals.size(); ++i) {
            // loss_fn takes (zeros, error) and returns a scalar loss
            // matrix zeros = matrix::Zero(residuals[i].rows(), residuals[i].cols());
            losses_val.push_back(loss_fn(residuals[i]));
        }
    }

    // Boundary condition errors
    // BC points in batch are arranged as: [BC0 | BC1 | ... | BCn]
    // where BCi occupies rows [bcs_start[i], bcs_start[i+1])
    for (size_t i = 0; i < bcs.size(); ++i) {
        
        if (bcs_start[i + 1] > inputs.rows()) {
            throw std::runtime_error("BC index range exceeds inputs/outputs rows");
        }

        // Compute BC error using the boundary condition object
        // bcs[i]->error expects: (train_x_all, inputs_batch, outputs_batch, beg, end)
        // where train_x_all is the full training set and beg, end index into the batch
        matrix bc_err = bcs[i]->error(inputs, inputs, outputs, bcs_start[i], bcs_start[i + 1]);
        
        losses_val.push_back(loss_fn(bc_err));
    }
    
    return losses_val;
}

matrix& Pde::train_next_batch(const int batch_size){

    if(batch_size == 0){
        // Full batch: use pre-allocated space (points already generated in constructor)
        if(!is_train_x_initialized){
            size_t index_beg = 0;
            for(size_t i = 0; i < train_x_bc.size(); ++i){
                train_x.block(index_beg, 0, num_bcs[i], phisical_dim) = train_x_bc[i];
                index_beg += num_bcs[i];
            }
            train_x.block(num_boundary, 0, train_x_pde.rows(), phisical_dim) = train_x_pde;
            is_train_x_initialized = true;
        }
    }
    else{
        // Start with BC points then append PDE points

        const int len_pde = train_x_pde.rows();
        const int len_training = train_x_pde.rows() + num_boundary;

        // Full batch if non-positive or larger than training set
        if (batch_size <= 0 || batch_size >= len_training) {
            training_batch_size = len_training;
            for(size_t i = 0; i< idx_bc_prec.size(); ++i){
                idx_bc_prec[i] = 0;
            }
            idx_pde_prec = 0;
            std::cerr << "Warning: The given batch size is more than the entire training set or negative. So the full set is used." << std::endl;
            return train_x;
        }
        
        bool reshuffle = false;

        // Compute batch split between BC and PDE portions
        // keeps the same proportion of the training set
        // moreover ensures that there is at least one point for every boundary condition
        int batch_size_bc = std::min( std::max(
                                                static_cast<int>(bcs.size()), 
                                                batch_size * num_boundary / len_training ), num_boundary);
        
        int batch_size_pde = std::max(0, std::min(batch_size - batch_size_bc,  
                                                    len_pde));

        // Build the batch: [BC batch; PDE batch]

        train_x.resize(batch_size_bc + batch_size_pde, phisical_dim);

        // assign the bc points
        const int got_bc = batch_size_bc / bcs.size();
        int residual_bc = batch_size_bc % bcs.size();
        size_t index = 0;
        for(size_t i = 0; i < train_x_bc.size(); ++i){
            num_bcs_batch[i] = std::min(got_bc + (residual_bc > 0 ), num_bcs[i] - idx_bc_prec[i]);
            residual_bc -= (residual_bc > 0);
            train_x.block(index, 0, num_bcs_batch[i], phisical_dim) = train_x_bc[i].block(idx_bc_prec[i], 0, num_bcs_batch[i], phisical_dim);
            idx_bc_prec[i] += num_bcs_batch[i];
            if(idx_bc_prec[i] >= num_bcs[i]){
                reshuffle = true;
            }
        }
        
        // assign domain points
        const int pde_end = std::min(idx_pde_prec + batch_size_pde, len_pde);
        const int got_pde = pde_end - idx_pde_prec;

        train_x.bottomRows(got_pde) = train_x_pde.block(idx_pde_prec, 0, got_pde, phisical_dim);

        idx_pde_prec = pde_end;

        // If we reached the end of either stream, reshuffle for next epoch and reset indices
        if (reshuffle || idx_pde_prec >= len_pde) {
            std::mt19937 rng(epoch_seed);

            // Permute BC set
            for(size_t i = 0; i < train_x_bc.size(); ++i){
                const int bc_rows = static_cast<int>(train_x_bc[i].rows());
                Eigen::VectorXi perm = Eigen::VectorXi::LinSpaced(bc_rows, 0, bc_rows - 1);
                std::shuffle(perm.data(), perm.data() + perm.size(), rng);
                matrix shuffled(train_x_bc[i].rows(), phisical_dim);

                for (size_t j = 0; j < bc_rows; ++j){
                    shuffled.row(j) = train_x_bc[i].row(perm[j]);
                }

                train_x_bc[i].swap(shuffled);
                idx_bc_prec[i] = 0;
            }
            // Permute PDE set
            Eigen::VectorXi perm = Eigen::VectorXi::LinSpaced(len_pde, 0, len_pde - 1);
            std::shuffle(perm.data(), perm.data() + perm.size(), rng);
            matrix shuffled(len_pde, train_x_pde.cols());
            for (int i = 0; i < len_pde; ++i) shuffled.row(i) = train_x_pde.row(perm[i]);
            train_x_pde.swap(shuffled);
            
            idx_pde_prec = 0;

        }

        training_batch_size = batch_size;
    }
    return train_x;
}