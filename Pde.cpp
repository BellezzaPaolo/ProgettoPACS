#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <random>
#include <cstddef>
#include <cstring>
#include <pybind11/numpy.h>
#include "Pde.hpp"

namespace { // anonymous namespace so this functions can only be used in this file
tensor pyarray_to_tensor_2d_double(const py::handle& obj) {

    // Forcecast converts in float64 numpy array while c_style ensures the data to be row-major
    // ordered. 
    // template<typename T>
    //  T reinterpret_borrow(handle h):
    //      Declare that a handle or PyObject * is a certain type and borrow the reference. 
    //      The target type T must be object or one of its derived classes. The function doesn’t do any conversions or checks.
    py::array_t<double, py::array::c_style | py::array::forcecast> arr =
        py::reinterpret_borrow<py::object>(obj).cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();

    // simple check of the dimensions
    py::buffer_info info = arr.request();
    if (info.ndim != 2) {
        throw std::runtime_error("Expected a 2D numpy array for geometry points");
    }

    // get dimensions
    const int64_t n = static_cast<int64_t>(info.shape[0]);
    const int64_t d = static_cast<int64_t>(info.shape[1]);

    tensor out = torch::empty({n, d}, torch::dtype(torch::kDouble).device(torch::kCPU));

    // void* memcpy( void* dest, const void* src, std::size_t count );
	//      Performs the following operations in order:
    //         - Implicitly creates objects at dest.
    //         - Copies count characters (as if of type unsigned char) from the object pointed to by src into the object pointed to by dest. 
    std::memcpy(out.data_ptr<double>(), info.ptr, static_cast<size_t>(n * d) * sizeof(double));
    return out;
}

tensor permute_rows_cpu_double(const tensor& X, std::mt19937& rng) {

    // some checks
    TORCH_CHECK(X.device().is_cpu(), "permute_rows_cpu_double expects CPU tensor");
    TORCH_CHECK(X.scalar_type() == torch::kDouble, "permute_rows_cpu_double expects double tensor");
    TORCH_CHECK(X.dim() == 2, "permute_rows_cpu_double expects 2D tensor");

    const int64_t n = X.size(0);
    if (n <= 1) {
        return X;
    }

    std::vector<int64_t> perm(static_cast<size_t>(n));
    // void iota( ForwardIt first, ForwardIt last, T value ):
    //      Fills the range [first, last) with sequentially increasing values, starting with
    //      value and repetitively evaluating ++value
    std::iota(perm.begin(), perm.end(), 0);
    // void shuffle( RandomIt first, RandomIt last, URBG&& g );
    std::shuffle(perm.begin(), perm.end(), rng);

    // convert the vector into torch tensor
    tensor idx = torch::from_blob(perm.data(), {n}, torch::TensorOptions().dtype(torch::kLong)).clone();
    // returns the selected index
    return X.index_select(0, idx);
}
} // namespace

Pde::Pde(py::handle geom,
    std::function<std::vector<tensor>(const tensor&, const tensor&)> pde,
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
        train_x_pde = torch::zeros({num_domain + num_boundary, phisical_dim}, torch::dtype(torch::kDouble));
        train_x_bc.resize(bcs.size());
        num_bcs.resize(bcs.size());
        
        // Generate training and BC points once
        generate_train_points();
        generate_bc_points();
        is_train_x_initialized = false;

        if(num_test > 0){
            test = torch::zeros({num_test + num_boundary, phisical_dim}, torch::dtype(torch::kDouble));
            generate_test_points();
        }

        // train_x is built dynamically (full set or batches)
        train_x = torch::empty({0, phisical_dim}, torch::dtype(torch::kDouble));

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

    tensor X = pyarray_to_tensor_2d_double(X_py);
    tensor X_bc = pyarray_to_tensor_2d_double(X_bc_py);
    TORCH_CHECK(X.size(1) == phisical_dim, "Geometry returned wrong dimension for domain points");
    TORCH_CHECK(X_bc.size(1) == phisical_dim, "Geometry returned wrong dimension for boundary points");

    // tensor.narrow(int64_t dim, int64_t t_start, int64_t length)
    //    returns a view of a tensor restricted to a contiguous slice along dim (0 = rows for a 2D tensor, 1 = columns) from t_start 
    //    for length elements
    train_x_pde.narrow(0, 0, num_domain).copy_(X);
    train_x_pde.narrow(0, num_domain, num_boundary).copy_(X_bc);

    std::cout << "train_x_pde: " << train_x_pde.size(0) << " " << train_x_pde.size(1) << std::endl;
    return;
}

void Pde::generate_bc_points(){
    int real_points_on_boundary = 0;
    for(size_t i = 0; i < bcs.size(); ++i){
        train_x_bc[i] = bcs[i]->collocation_points(train_x_pde);

        num_bcs[i] = static_cast<int>(train_x_bc[i].size(0));

        real_points_on_boundary += num_bcs[i];

        std::cout << " bc "<< i << " num_bcs: " << num_bcs[i] << " train_x_bc[i] " << train_x_bc[i] <<std::endl;
    }
    // reassign num_boundary
    num_boundary = real_points_on_boundary;
    std::cout << " num_boundary " << num_boundary << std::endl;
    return;
}

void Pde::generate_test_points(){
    if(num_test > 0){
        // the distribution here is random beacuse for same geometries the "uniform_point" function is not implemented
        py::object X_py = geom.attr("random_points")(num_test);
        tensor X = pyarray_to_tensor_2d_double(X_py);
        TORCH_CHECK(X.size(1) == phisical_dim, "Geometry returned wrong dimension for test points");

        // tensor.narrow(int64_t dim, int64_t t_start, int64_t length)
        //    returns a view of a tensor restricted to a contiguous slice along dim (0 = rows for a 2D tensor, 1 = columns) from t_start 
        //    for length elements
        test.narrow(0, 0, num_test).copy_(X);

        int64_t row = num_test;
        for(const tensor& elem_bc : train_x_bc){
            const int64_t r = elem_bc.size(0);
            if (r == 0) {
                continue;
            }
            TORCH_CHECK(row + r <= test.size(0), "Test tensor is too small for boundary points");

            // tensor.narrow(int64_t dim, int64_t t_start, int64_t length)
            //    returns a view of a tensor restricted to a contiguous slice along dim (0 = rows for a 2D tensor, 1 = columns) from t_start 
            //    for length elements
            test.narrow(0, row, r).copy_(elem_bc.to(torch::kCPU));
            row += r;
        }
    }
    else{
        throw std::runtime_error("num_test is zero so no test set");
    }
    return;
}

std::vector<double> Pde::losses(
    const tensor& inputs,
    const tensor& outputs,
    const std::function<double(const tensor&)>& loss_fn
){
    // Validate shapes
    TORCH_CHECK(inputs.dim() == 2, "inputs must be 2D [N, dim]");
    TORCH_CHECK(outputs.dim() == 2, "outputs must be 2D [N, out_dim]");
    if (inputs.size(0) != outputs.size(0)) {
        std::cout << "inputs rows are "<< inputs.size(0) << " and outputs rows are "<< outputs.size(0)<< std::endl; 
        throw std::runtime_error("inputs and outputs must have the same number of rows");
    }

    // Choose BC counts: use batch-specific if available and matching this inputs size
    std::vector<int> nb;
    if (static_cast<int>(inputs.size(0)) == training_batch_size) {
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
    if (total_bc > inputs.size(0)) {
        throw std::runtime_error("num_bcs exceeds provided inputs/outputs rows");
    }

    // Collect all loss values (PDE losses first, then BC losses)
    std::vector<double> losses_val;

    // PDE residuals on the remaining rows (after all BC points)
    const int pde_rows = static_cast<int>(inputs.size(0)) - total_bc;
    std::vector<tensor> residuals;
    
    if (pde_rows > 0) {
        // User-defined residual(s): vector of matrices

        // tensor.narrow(int64_t dim, int64_t t_start, int64_t length)
        //    returns a view of a tensor restricted to a contiguous slice along dim (0 = rows for a 2D tensor, 1 = columns) from t_start 
        //    for length elements
        // tensor inputs_pde = inputs.narrow(0, total_bc, pde_rows);
        // tensor outputs_pde = outputs.narrow(0, total_bc, pde_rows);
        residuals = pde(inputs.narrow(0, total_bc, pde_rows), outputs.narrow(0, total_bc, pde_rows));
        
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
        
        if (bcs_start[i + 1] > inputs.size(0)) {
            throw std::runtime_error("BC index range exceeds inputs/outputs rows");
        }

        // Compute BC error using the boundary condition object
        // bcs[i]->error expects: (train_x_all, inputs_batch, outputs_batch, beg, end)
        // where train_x_all is the full training set and beg, end index into the batch
        tensor bc_err = bcs[i]->error(train_x, inputs, outputs, bcs_start[i], bcs_start[i + 1]);
        
        losses_val.push_back(loss_fn(bc_err));
    }
    
    return losses_val;
}

tensor& Pde::train_next_batch(const int batch_size){

    if(batch_size == 0){
        // Full batch: cache concatenation
        if(!is_train_x_initialized){
            // Build once into a contiguous buffer: [BC0 | BC1 | ... | PDE]
            int64_t total_bc = 0;
            for (size_t i = 0; i < train_x_bc.size(); ++i) {
                if (train_x_bc[i].defined()) {
                    total_bc += train_x_bc[i].size(0);
                }
            }

            const int64_t total_rows = total_bc + train_x_pde.size(0);
            train_x = train_x.resize_({total_rows, phisical_dim});

            int64_t offset = 0;
            for (size_t i = 0; i < train_x_bc.size(); ++i) {
                if (!train_x_bc[i].defined() || train_x_bc[i].numel() == 0) {
                    continue;
                }
                const int64_t r = train_x_bc[i].size(0);

                // tensor.narrow(int64_t dim, int64_t t_start, int64_t length)
                //    returns a view of a tensor restricted to a contiguous slice along dim (0 = rows for a 2D tensor, 1 = columns) from t_start 
                //    for length elements
                train_x.narrow(0, offset, r).copy_(train_x_bc[i]);
                offset += r;
            }
            // Append PDE points
            if (train_x_pde.numel() > 0) {
                const int64_t r = train_x_pde.size(0);
                train_x.narrow(0, offset, r).copy_(train_x_pde);
                offset += r;
            }

            TORCH_CHECK(offset == total_rows, "Internal error: full batch assembly size mismatch");
            is_train_x_initialized = true;
        }
    }
    else{
        // Start with BC points then append PDE points

        const int len_pde = static_cast<int>(train_x_pde.size(0));
        const int len_training = len_pde + num_boundary;

        // Full batch if non-positive or larger than training set
        if (batch_size <= 0 || batch_size >= len_training) {
            training_batch_size = len_training;
            for(size_t i = 0; i< idx_bc_prec.size(); ++i){
                idx_bc_prec[i] = 0;
            }
            idx_pde_prec = 0;
            std::cerr << "Warning: The given batch size is more than the entire training set or negative. So the full set is used." << std::endl;
            return train_next_batch(/*batch_size=*/0);
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

        // Build the batch: [BC batch; PDE batch] into a reusable contiguous buffer

        // assign the bc points
        const int got_bc = batch_size_bc / bcs.size();
        int residual_bc = batch_size_bc % bcs.size();
        int64_t actual_bc = 0;
        for(size_t i = 0; i < train_x_bc.size(); ++i){
            num_bcs_batch[i] = std::min(got_bc + (residual_bc > 0 ), num_bcs[i] - idx_bc_prec[i]);
            residual_bc -= (residual_bc > 0);
            actual_bc += num_bcs_batch[i];
            if(idx_bc_prec[i] >= num_bcs[i]){
                reshuffle = true;
            }
        }
        
        // assign domain points
        const int pde_end = std::min(idx_pde_prec + batch_size_pde, len_pde);
        const int got_pde = pde_end - idx_pde_prec;

        const int64_t total_rows = actual_bc + static_cast<int64_t>(got_pde);
        train_x = train_x.resize_({total_rows, phisical_dim});

        int64_t offset = 0;
        // Rewind BC cursors by the amount we just advanced so we can copy slices
        for (size_t i = 0; i < train_x_bc.size(); ++i) {
            const int k = num_bcs_batch[i];
            if (k <= 0) {
                std::cout << "Warning! the boundary condition " << i << " has no points in this batch" <<std::endl;
                continue;
            }
            const int start = idx_bc_prec[i];
            train_x.narrow(0, offset, k).copy_(train_x_bc[i].narrow(0, start, k));
            offset += k;
            idx_bc_prec[i] += num_bcs_batch[i];
        }

        if (got_pde > 0) {
            train_x.narrow(0, offset, got_pde).copy_(train_x_pde.narrow(0, idx_pde_prec, got_pde));
            offset += got_pde;
        }

        TORCH_CHECK(offset == total_rows, "Internal error: batch assembly size mismatch");

        idx_pde_prec = pde_end;

        // If we reached the end of either stream, reshuffle for next epoch and reset indices
        if (reshuffle || idx_pde_prec >= len_pde) {
            std::mt19937 rng(epoch_seed);

            // Permute BC set
            for(size_t i = 0; i < train_x_bc.size(); ++i){
                train_x_bc[i] = permute_rows_cpu_double(train_x_bc[i].to(torch::kCPU).contiguous(), rng);
                idx_bc_prec[i] = 0;
            }
            // Permute PDE set
            train_x_pde = permute_rows_cpu_double(train_x_pde.to(torch::kCPU).contiguous(), rng);
            
            idx_pde_prec = 0;

        }

        training_batch_size = static_cast<int>(train_x.size(0));
    }
    return train_x;
}