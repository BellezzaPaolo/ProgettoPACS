#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <random>
#include <cstddef>
#include <cstring>
#include <pybind11/numpy.h>
#include "Pde.hpp"

/**
 * @file Pde.cpp
 * @brief Implementation of the `Pde` data container.
 *
 * This file implements:
 * - conversion utilities from NumPy arrays to CPU float32 tensors
 * - point generation via DeepXDE geometry (Python)
 * - loss term assembly (PDE + boundary conditions)
 * - batch assembly/shuffling
 */

/**
 * @brief Internal helpers (file-local).
 * @details Placed in an anonymous namespace so they are only visible in this translation unit.
 */
namespace {
/**
 * @brief Convert a Python NumPy array to a 2D CPU tensor (float32, C-contiguous).
 * @param obj Python object convertible to a NumPy array.
 * @return Tensor of shape `[N, D]` with dtype float32 on CPU.
 *
 * @throws std::runtime_error if `obj` is not 2D.
 */
tensor pyarray_to_tensor_2d_float(const py::handle& obj) {
    /**
     * @details
     * `forcecast` converts input to float32 when needed.
     * `c_style` ensures row-major contiguous layout.
     */
    py::array_t<float, py::array::c_style | py::array::forcecast> arr =
        py::reinterpret_borrow<py::object>(obj).cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

    /** @details Validate that the NumPy array is 2D. */
    py::buffer_info info = arr.request();
    if (info.ndim != 2) {
        throw std::runtime_error("Expected a 2D numpy array for geometry points");
    }

    /** @details Read array dimensions. */
    const int64_t n = static_cast<int64_t>(info.shape[0]);
    const int64_t d = static_cast<int64_t>(info.shape[1]);

    tensor out = torch::empty({n, d}, torch::dtype(torch::kFloat32).device(torch::kCPU));

    /** @details Fast contiguous copy from NumPy buffer into the torch tensor. */
    std::memcpy(out.data_ptr<float>(), info.ptr, static_cast<size_t>(n * d) * sizeof(float));
    return out;
}

/**
 * @brief Shuffle rows of a 2D CPU tensor.
 * @param X Input tensor `[N, D]` on CPU.
 * @param rng Random number generator used to shuffle indices.
 * @return Shuffled copy of `X`.
 */
tensor permute_rows_cpu(const tensor& X, std::mt19937& rng) {
    /** @details Validate device/shape assumptions. */
    TORCH_CHECK(X.device().is_cpu(), "permute_rows_cpu expects CPU tensor");
    TORCH_CHECK(X.dim() == 2, "permute_rows_cpu expects 2D tensor");

    const int64_t n = X.size(0);
    if (n <= 1) {
        return X;
    }

    std::vector<int64_t> perm(static_cast<size_t>(n));
    /** @details Build an index permutation and shuffle it. */
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    /** @details Convert the permutation into an index tensor. */
    tensor idx = torch::from_blob(perm.data(), {n}, torch::TensorOptions().dtype(torch::kLong)).clone();
    /** @details Return the row-permuted tensor. */
    return X.index_select(0, idx);
}
} // namespace

/**
 * @brief Construct dataset, sample training/test points, and initialize batch state.
 */
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
        /** @details Initialize structures to store the training set. */
        train_x_pde = torch::zeros({num_domain + num_boundary, phisical_dim}, torch::dtype(torch::kFloat32));
        train_x_bc.resize(bcs.size());
        num_bcs.resize(bcs.size());
        
        /** @details Generate training and boundary-condition points once. */
        generate_train_points();
        generate_bc_points();
        is_train_x_initialized = false;

        if(num_test > 0){
            test = torch::zeros({num_test + num_boundary, phisical_dim}, torch::dtype(torch::kFloat32));
            generate_test_points();
        }

        /** @details `train_x` is built dynamically (full set or batches). */
        train_x = torch::empty({0, phisical_dim}, torch::dtype(torch::kFloat32));

        idx_pde_prec = 0;
        idx_bc_prec.resize(bcs.size());
        epoch_seed = 123;
        /** @details Sizes for the current batch BC splits. */
        num_bcs_batch.resize(bcs.size());
        training_batch_size = 0;
    };

void Pde::generate_train_points(){
    /** @brief Generate interior and boundary points using the DeepXDE geometry. */
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

    tensor X = pyarray_to_tensor_2d_float(X_py);
    tensor X_bc = pyarray_to_tensor_2d_float(X_bc_py);
    TORCH_CHECK(X.size(1) == phisical_dim, "Geometry returned wrong dimension for domain points");
    TORCH_CHECK(X_bc.size(1) == phisical_dim, "Geometry returned wrong dimension for boundary points");

    /** @details Copy into the preallocated `train_x_pde` buffer via `narrow()` views. */
    train_x_pde.narrow(0, 0, num_domain).copy_(X);
    train_x_pde.narrow(0, num_domain, num_boundary).copy_(X_bc);

    return;
}

void Pde::generate_bc_points(){
    /** @brief Generate collocation points for each boundary condition. */
    int real_points_on_boundary = 0;
    for(size_t i = 0; i < bcs.size(); ++i){
        train_x_bc[i] = bcs[i]->collocation_points(train_x_pde);

        num_bcs[i] = static_cast<int>(train_x_bc[i].size(0));

        real_points_on_boundary += num_bcs[i];

    }
    /** @details Reassign `num_boundary` to the actual count after filtering. */
    num_boundary = real_points_on_boundary;
    return;
}

void Pde::generate_test_points(){
    /** @brief Generate test points (interior + boundary) used for monitoring generalization. */
    if(num_test > 0){
        /** @details Use random distribution: some geometries do not implement uniform sampling. */
        py::object X_py = geom.attr("random_points")(num_test);
        tensor X = pyarray_to_tensor_2d_float(X_py);
        TORCH_CHECK(X.size(1) == phisical_dim, "Geometry returned wrong dimension for test points");

        /** @details Copy interior test points into the first block. */
        test.narrow(0, 0, num_test).copy_(X);

        int64_t row = num_test;
        for(const tensor& elem_bc : train_x_bc){
            const int64_t r = elem_bc.size(0);
            if (r == 0) {
                continue;
            }
            TORCH_CHECK(row + r <= test.size(0), "Test tensor is too small for boundary points");

            /** @details Append BC points after the interior points. */
            test.narrow(0, row, r).copy_(elem_bc.to(torch::kCPU));
            row += r;
        }
    }
    else{
        throw std::runtime_error("num_test is zero so no test set");
    }
    return;
}

std::vector<tensor> Pde::losses(
    const tensor& inputs,
    const tensor& outputs,
    const std::function<tensor(const tensor&)>& loss_fn
){
    /**
     * @details
     * Returns differentiable scalar tensors that can be summed and backpropagated.
     * Layout: PDE losses first, then BC losses.
     */
    TORCH_CHECK(inputs.dim() == 2, "inputs must be 2D [N, dim]");
    TORCH_CHECK(outputs.dim() == 2, "outputs must be 2D [N, out_dim]");
    if (inputs.size(0) != outputs.size(0)) {
        std::cout << "inputs rows are "<< inputs.size(0) << " and outputs rows are "<< outputs.size(0)<< std::endl; 
        throw std::runtime_error("inputs and outputs must have the same number of rows");
    }

    /**
     * @details
     * Choose BC counts: use batch-specific counts when `inputs` correspond to the
     * most recently built batch, otherwise fall back to full-dataset counts.
     */
    std::vector<int> nb;
    if (static_cast<int>(inputs.size(0)) == training_batch_size) {
        nb = num_bcs_batch;
    } else {
        nb = num_bcs;
    }

    /**
     * @details
     * Compute cumulative BC indices: `bcs_start[i]` is the starting row for BC i.
     * Batch structure is:
     * `[BC0_points | BC1_points | ... | BCn_points | PDE_points]`.
     */

    std::vector<int> bcs_start;
    bcs_start.push_back(0);
    for (size_t i = 0; i < nb.size(); ++i) {
        bcs_start.push_back(bcs_start.back() + nb[i]);
    }
    
    const int total_bc = bcs_start.back();
    if (total_bc > inputs.size(0)) {
        throw std::runtime_error("num_bcs exceeds provided inputs/outputs rows");
    }

    /** @details Collect all loss values (PDE losses first, then BC losses). */
    std::vector<tensor> losses_val;

    /** @details Scalar 0 on the right device/dtype for empty segments. */
    tensor zero = torch::zeros({}, outputs.options());

    /** @details PDE residuals live on the remaining rows (after all BC points). */
    const int pde_rows = static_cast<int>(inputs.size(0)) - total_bc;

    /**
     * @details
     * Important: call the PDE callback on the *full* `(inputs, outputs)` tensors.
     * This guarantees autograd sees a direct dependency between inputs and outputs.
     * Passing sliced views can sometimes trigger "unused" gradients.
     */
    std::vector<tensor> residuals_full = pde(inputs, outputs);

    for (size_t i = 0; i < residuals_full.size(); ++i) {
        TORCH_CHECK(residuals_full[i].dim() == 2, "PDE residual must be 2D [N, n_res]");
        TORCH_CHECK(residuals_full[i].size(0) == inputs.size(0), "PDE residual rows must match inputs rows");

        if (pde_rows <= 0) {
            losses_val.push_back(zero);
            std::cout << "Warning! the number of rows realted to the pde are 0 or less" << std::endl;
            continue;
        }

        tensor r_pde = residuals_full[i].narrow(0, total_bc, pde_rows);
        losses_val.push_back(loss_fn(r_pde));
    }
    

    /**
     * @details
     * Boundary condition errors.
     * BC points in batch are arranged as `[BC0 | BC1 | ... | BCn]` where BC i
     * occupies rows `[bcs_start[i], bcs_start[i+1])`.
     */
    for (size_t i = 0; i < bcs.size(); ++i) {
        
        if (bcs_start[i + 1] > inputs.size(0)) {
            throw std::runtime_error("BC index range exceeds inputs/outputs rows");
        }

        /**
         * @details
         * Compute BC error using the boundary condition object.
         * The signature is `error(train_x_all, inputs_batch, outputs_batch, beg, end)`.
         */
        const int beg = bcs_start[i];
        const int end = bcs_start[i + 1];
        if (end <= beg) {
            std::cout << "Warning! the number of rows realted to the boundary condition " << i <<" are 0 or less" << std::endl;
            losses_val.push_back(zero);
            continue;
        }

        tensor bc_err = bcs[i]->error(train_x, inputs, outputs, beg, end);

        losses_val.push_back(loss_fn(bc_err));
    }
    
    return losses_val;
}

tensor& Pde::train_next_batch(const int batch_size){
    /**
     * @details
     * Returns a contiguous tensor of training points.
     * Batch layout is consistent with `losses()`:
     * `[BC0 | BC1 | ... | PDE]`.
     */

    if(batch_size == 0){
        /** @details Full batch: cache concatenation. */
        if(!is_train_x_initialized){
            /** @details Build once into a contiguous buffer: `[BC0 | BC1 | ... | PDE]`. */
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

                /** @details Copy via `narrow()` view to avoid extra allocations. */
                train_x.narrow(0, offset, r).copy_(train_x_bc[i]);
                offset += r;
            }
            /** @details Append PDE points. */
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
        /** @details Start with BC points then append PDE points. */

        const int len_pde = static_cast<int>(train_x_pde.size(0));
        const int len_training = len_pde + num_boundary;

        /** @details Full batch if non-positive or larger than training set. */
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

        /**
         * @details
         * Compute batch split between BC and PDE portions:
         * keep approximately the same proportion as the training set and ensure
         * at least one point per boundary condition.
         */
        int batch_size_bc = std::min( std::max(
                                                static_cast<int>(bcs.size()), 
                                                batch_size * num_boundary / len_training ), num_boundary);
        
        int batch_size_pde = std::max(0, std::min(batch_size - batch_size_bc,  
                                                    len_pde));

        /** @details Build the batch: `[BC batch; PDE batch]` into a contiguous buffer. */

        /** @details Assign the BC points. */
        const int got_bc = batch_size_bc / bcs.size();
        int residual_bc = batch_size_bc % bcs.size();
        int64_t actual_bc = 0;
        for(size_t i = 0; i < train_x_bc.size(); ++i){
            num_bcs_batch[i] = std::min(got_bc + (residual_bc > 0 ), num_bcs[i] - idx_bc_prec[i]);
            residual_bc -= (residual_bc > 0);
            actual_bc += num_bcs_batch[i];
            if(idx_bc_prec[i] + num_bcs_batch[i] >= num_bcs[i]){
                reshuffle = true;
            }
        }
        
        /** @details Assign domain (PDE) points. */
        const int pde_end = std::min(idx_pde_prec + batch_size_pde, len_pde);
        const int got_pde = pde_end - idx_pde_prec;

        const int64_t total_rows = actual_bc + static_cast<int64_t>(got_pde);

        is_train_x_initialized = false;
        
        train_x = train_x.resize_({total_rows, phisical_dim});

        int64_t offset = 0;
        /** @details Copy the selected BC slices into the contiguous batch buffer. */
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

        /** @details If we reached the end, reshuffle for next epoch and reset indices. */
        if (reshuffle || idx_pde_prec >= len_pde) {
            std::mt19937 rng(epoch_seed);

            /** @details Permute BC sets. */
            for(size_t i = 0; i < train_x_bc.size(); ++i){
                train_x_bc[i] = permute_rows_cpu(train_x_bc[i].to(torch::kCPU).contiguous(), rng);
                idx_bc_prec[i] = 0;
            }
            /** @details Permute PDE set. */
            train_x_pde = permute_rows_cpu(train_x_pde.to(torch::kCPU).contiguous(), rng);
            
            idx_pde_prec = 0;

        }

        training_batch_size = static_cast<int>(train_x.size(0));
    }
    return train_x;
}