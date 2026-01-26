#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "Param.hpp"

// Activation functors (stateless) kept separate for clarity.
struct ReluAct {
    template <typename T>
    vector_t<T> operator () (const vector_t<T>& x) const {
        return x.array().max(static_cast<T>(0)).matrix();
    }
};

struct LinearAct {
    template <typename T>
    vector_t<T> operator () (const vector_t<T>& x) const {
        return x;
    }
};

struct TanhAct {
    template <typename T>
    vector_t<T> operator () (const vector_t<T>& x) const {
        return x.array().tanh().matrix();
    }
};

struct SigmoidAct {
    template <typename T>
    vector_t<T> operator () (const vector_t<T>& x) const {
        return (static_cast<T>(1) / (static_cast<T>(1) + (-x.array()).exp())).matrix();
    }
};

#endif
