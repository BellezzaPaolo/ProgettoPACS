#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "Param.hpp"

// Activation functors (stateless) kept separate for clarity.
struct ReluAct {
    template <typename T>
    static vector_t<T> apply(const vector_t<T>& x) {
        return x.array().max(static_cast<T>(0)).matrix();
    }
};

struct LinearAct {
    template <typename T>
    static vector_t<T> apply(const vector_t<T>& x) {
        return x;
    }
};

struct TanhAct {
    template <typename T>
    static vector_t<T> apply(const vector_t<T>& x) {
        return x.array().tanh().matrix();
    }
};

struct SigmoidAct {
    template <typename T>
    static vector_t<T> apply(const vector_t<T>& x) {
        return (static_cast<T>(1) / (static_cast<T>(1) + (-x.array()).exp())).matrix();
    }
};

#endif
