#ifndef PDE_HPP
#define PDE_HPP

/**
 * @file Pde.hpp
 * @brief Dataset/geometry wrapper for PINNs (DeepXDE-like).
 *
 * This class owns:
 * - a DeepXDE geometry handle (Python object)
 * - collocation points for PDE residuals and boundary conditions
 * - batching/shuffling logic
 * - a user-provided PDE residual callback evaluated with LibTorch tensors
 *
 * The PDE itself is defined in C++ (typically as a lambda in `main.cpp`) and can
 * use torch autograd to build residuals such as Laplacians.
 */
#include <vector>
#include <memory>
#include <functional>
#include <pybind11/embed.h>
#include <torch/torch.h>
#include "boundary_condition/Boundary_Condition.hpp"

namespace py = pybind11;

/// Alias used across the project for LibTorch tensors.
using tensor = torch::Tensor;

/// \cond DOXYGEN_SHOULD_SKIP_THIS
// These pragmas silence GCC's -Wattributes visibility warnings. See the note at
// the end of this file for details.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif
/// \endcond

/**
 * @brief PDE + BC data container and batching utility.
 *
 * @details
 * This class mimics DeepXDE's `data.PDE` role: it stores training/test points
 * and provides a `losses()` method that returns differentiable scalar tensors.
 *
 * Training batches are assembled in the following row order:
 * @code
 * [ BC0 | BC1 | ... | BC(k-1) | PDE ]
 * @endcode
 * so the first rows are reserved for boundary-condition points (grouped by BC)
 * and the remaining rows are domain points for the PDE residual.
 *
 * @note
 * The geometry and boundary-condition objects are backed by Python (pybind11).
 * This means the Python interpreter must be initialized before constructing
 * a `Pde` instance.
 */

class Pde {
private:
    // Geometry interface (Python: geom)
    py::handle geom;
    int phisical_dim;

    // PDE residual function
    std::function<std::vector<tensor>(const tensor& inputs, const tensor& outputs)> pde;

    // Boundary conditions
    std::vector<std::shared_ptr<Boundary_Condition>> bcs;

    int num_domain;
    int num_boundary;
    int num_test;

    std::string train_distribution;

    tensor train_x_pde; // is train_x_all of deepXDE, following the TODO comment has been changed the name
    std::vector<tensor> train_x_bc;
    tensor test;

    std::vector<int> num_bcs;

    // TODO add the exact solution and its handle in the generation of the training and test set 

    tensor train_x;
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
    /**
     * @brief Construct the dataset and pre-generate training/test points.
     *
     * @param geom DeepXDE geometry object (Python handle).
     * @param pde PDE residual callback. It must return a vector of residual tensors
     *            with shape `[N, r_i]` where `N` is the number of input rows.
     * @param bcs Boundary condition objects.
     * @param Num_domain Number of interior/domain points requested.
     * @param Num_boundary Number of boundary points requested.
     * @param Num_test Number of test points requested (0 disables test set).
     * @param train_distribution Distribution string forwarded to DeepXDE sampling.
     */
    Pde(py::handle geom,
        std::function<std::vector<tensor>(const tensor&, const tensor&)> pde,
        std::vector<std::shared_ptr<Boundary_Condition>> bcs,
        int Num_domain, int Num_boundary, int Num_test = 0, std::string train_distribution = "Hammersley");
    
    /**
     * @brief Get the test set points.
     * @return Reference to a tensor of shape `[N_test, dim]`.
     */
    const tensor& get_test() const { return test; }

    /**
     * @brief Get the number of training points.
     * @return Reference to a tensor of shape `[N_test, dim]`.
     */
    bool has_test_set() const { return num_test > 0; }
    
    /**
     * @brief Compute differentiable scalar loss terms.
     *
     * @details
     * Returns a vector of scalar tensors:
     * - PDE loss terms first (one per PDE residual component returned by `pde`)
     * - then one loss term per boundary condition
     *
     * The caller is expected to aggregate them (e.g. `torch::stack(losses).sum()`).
     *
     * @param inputs Input coordinates `[N, dim]`.
     * @param outputs Network outputs `[N, out_dim]` evaluated at `inputs`.
     * @param loss_fn Loss function applied to residual/error tensors.
     * @return Vector of scalar tensors.
     */
    std::vector<tensor> losses(
        const tensor& inputs,
        const tensor& outputs,
        const std::function<tensor(const tensor&)>& loss_fn
    );

    /**
     * @brief Return the next training batch.
     *
     * @details
     * If `batch_size == 0`, this returns the full training set (cached after the
     * first assembly). If `batch_size > 0`, it builds a batch that keeps BC/PDE
     * proportions and ensures at least one point per BC when possible.
     *
     * @param batch_size Desired batch size. Use 0 for full batch.
     * @return Reference to an internal contiguous tensor `[N_batch, dim]`.
     */
    tensor& train_next_batch(const int batch_size = 0);
};

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/**
 * @note Visibility warning (-Wattributes)
 * Some toolchains compile third-party headers (e.g. pybind11) with hidden symbol
 * visibility. Since `Pde` stores pybind11 types (e.g. `py::handle`) and possibly
 * other types with different visibility, GCC may warn that `Pde` has “greater
 * visibility” than the type of one of its fields.
 *
 * This matters mainly when exporting `Pde` as part of a shared-library ABI.
 * In this project the executable is the main artifact, so we silence this
 * warning locally to keep build output clean.
 */
// The warning was:
// In file included from ./Desktop/ProgettoPACS/main.cpp:20:
// ./ProgettoPACS/Pde.hpp:14:7: warning: ‘Pde’ declared with greater visibility than the type of its field ‘Pde::geom’ [-Wattributes]
//    57 | class Pde {
//       |       ^~~
// ./ProgettoPACS/Pde.hpp:14:7: warning: ‘Pde’ declared with greater visibility than the type of its field ‘Pde::bcs’ [-Wattributes]

#endif

// #pragma once

// #include <vector>
// #include <memory>
// #include <functional>
// #include <pybind11/embed.h>
// #include <torch/torch.h>
// #include "boundary_condition/Boundary_Condition.hpp"

// namespace py = pybind11;

// using tensor = torch::Tensor;

// class __attribute__((visibility("hidden"))) Pde {
// private:
//     // Geometry interface (Python: geom)
//     py::handle geom;
//     int phisical_dim;

//     // PDE residual function
//     std::function<std::vector<tensor>(const tensor& inputs, const tensor& outputs)> pde;

//     // Boundary conditions
//     std::vector<std::shared_ptr<Boundary_Condition>> bcs;

//     int num_domain;
//     int num_boundary;
//     int num_test;

//     std::string train_distribution;

//     tensor train_x_pde; // is train_x_all of deepXDE, following the TODO comment has been changed the name
//     std::vector<tensor> train_x_bc;
//     tensor test;

//     std::vector<int> num_bcs;

//     // TODO add the exact solution and its handle in the generation of the training and test set 

//     tensor train_x;
//     bool is_train_x_initialized;

//     // Batching state
//     int idx_pde_prec;
//     std::vector<int> idx_bc_prec;
//     unsigned int epoch_seed = 123;
//     std::vector<int> num_bcs_batch; // sizes for current batch BC splits
//     int training_batch_size;

//     // this functions are made private because the generation of points in the domain
//     // can also take points on the boundary so it's not sure that train_x_bcs is exactly of 
//     // length num_boundary. So by now are accessible only through the constructor.
//     // TODO: make a function to regenerate teh point and consider resize test and train_x_bc.
//     void generate_train_points();

//     void generate_bc_points();

//     void generate_test_points();
// public:
//     // Constructor
//     Pde(py::handle geom,
//         std::function<std::vector<tensor>(const tensor&, const tensor&)> pde,
//         std::vector<std::shared_ptr<Boundary_Condition>> bcs,
//         int Num_domain, int Num_boundary, int Num_test = 4, std::string train_distribution = "Hammersley");

//     std::vector<double> losses(
//         const tensor& inputs,
//         const tensor& outputs,
//         const std::function<double(const tensor&)>& loss_fn
//     );

//     // Differentiable loss variant (returns scalar tensors you can backprop through).
//     std::vector<tensor> losses(
//         const tensor& inputs,
//         const tensor& outputs,
//         const std::function<tensor(const tensor&)>& loss_fn
//     );

//     // Build next training batch; if batch_size <= 0, use full set
//     tensor& train_next_batch(const int batch_size = 0);

//     // Number of boundary-condition points in the most recently produced batch.
//     // This is useful when you need to split a batch into [BC | PDE] parts.
//     int total_bc_in_last_batch() const;
// };
