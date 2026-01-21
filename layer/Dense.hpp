#ifndef DENSE_HPP
#define DENSE_HPP

#include "Param.hpp"
#include "activation.hpp"
#include <algorithm>

// Simple dense layer stored by value, inheriting from Param<T>
template <typename T, typename Activation>
class Dense : public Param<T> {
public:
    Dense(int in, int out) : Param<T>(in, out) {}

    template <Initializer_weight Iw, Initializer_bias Ib>
    void initialize() {
        this->template initialize_weight<Iw>();
        this->template initialize_bias<Ib>();
    }

    // Linear part only (no activation)
    vector_t<T> linear(const vector_t<T>& x) const {
        return (this->weights * x + this->bias).eval();
    }

    // // Accessors (read-only)
    // const matrix_t<T>& weight_matrix() const { return this->weights; }
    // const vector_t<T>& bias_vector() const { return this->bias; }

    vector_t<T> operator()(const vector_t<T>& x) const {
        return Activation::template apply<T>((this->weights * x + this->bias).eval());
    }

    void print() const{
        std::cout << "W:\n" << this-> weights << std::endl;
        std::cout << "bias:\n" << this-> bias << std::endl;
    }
};

#endif