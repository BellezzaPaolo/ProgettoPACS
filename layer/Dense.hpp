#ifndef DENSE_HPP
#define DENSE_HPP

#include "Param.hpp"
#include "activation.hpp"
#include <algorithm>

// Simple dense layer
template <typename T>
class Dense : public Param<T> {
public:
    Dense(int in, int out) : Param<T>(in, out) {}

    template <Initializer_weight Iw, Initializer_bias Ib>
    void initialize() {
        this->template initialize_weight<Iw>();
        this->template initialize_bias<Ib>();
    }

    // Linear part of forward pass
    vector_t<T> forward(const vector_t<T>& x) const {
        return (this->weights * x + this->bias).eval();
    }

    // Mixed-scalar forward pass: allows input scalar type to differ from
    // the stored parameter scalar type (e.g. weights/bias as double, input as
    // autodiff::var or autodiff::dual2nd).
    template <typename S>
    vector_t<S> forward(const vector_t<S>& x) const {
        // NOTE: do NOT cast weights/bias to S for reverse-mode autodiff::var.
        // Casting would turn every weight into an independent AD variable and can
        // explode the graph or break Hessian computation. Instead, rely on scalar
        // ops between (double) parameters and (S) activations.
        vector_t<S> y(this->weights.rows());
        for(int r = 0; r < this->weights.rows(); ++r){
            // Seed the accumulation from x to avoid introducing new leaf AD vars.
            S sum = x(0) * 0.0;
            for(int c = 0; c < this->weights.cols(); ++c){
                sum += this->weights(r, c) * x(c);
            }
            sum += this->bias(r);
            y(r) = sum;
        }
        return y;
    }

    // // Accessors (read-only)
    // const matrix_t<T>& weight_matrix() const { return this->weights; }
    // const vector_t<T>& bias_vector() const { return this->bias; }

    void print() const{
        std::cout << "W:\n" << this-> weights << std::endl;
        std::cout << "bias:\n" << this-> bias << std::endl;
    }
};

#endif