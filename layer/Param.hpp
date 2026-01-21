#ifndef PARAM_HPP
#define PARAM_HPP

#include <iostream>
#include <Eigen/Dense>
#include <random>


enum class Initializer_weight {One, Uniform, Glorot_Uniform, He_Uniform, Glorot_Norm, He_Norm};
enum class Initializer_bias {One, Zero, Uniform, Normal};

// Scalar-templated aliases: later we will instantiate with autodiff types.
template <typename T>
using matrix_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using vector_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;

// Backward-compatibility aliases for existing double-based code.
using matrix = matrix_t<double>;
using vector = vector_t<double>;

// namespace layer{

template <typename T>
class Param{
    protected:
        const int N_input;
        const int N_output;

        matrix_t<T> weights;
        vector_t<T> bias;

        template <typename U> friend class Param; // allow access across instantiations if needed

    public:
        Param(int n_input, int n_output): N_input(n_input), N_output(n_output){
            // initialize the weights and biases in a default mode for robustness
            initialize_weight<Initializer_weight::Glorot_Uniform>();
            initialize_bias<Initializer_bias::Zero>();
        };

        template <Initializer_weight Iw>
        void initialize_weight();

        template <Initializer_bias Ib>
        void initialize_bias();
};

// Convenience alias for current double-based usage
using Paramd = Param<double>;

// Inline template implementations
template <typename T>
template <Initializer_weight Iw>
inline void Param<T>::initialize_weight(){
    if constexpr (Iw == Initializer_weight::One){
        weights = matrix_t<T>::Ones(N_output,N_input);
    }
    else if constexpr (Iw == Initializer_weight::Uniform){
        weights = matrix_t<T>::Random(N_output,N_input);
    }
    else if constexpr (Iw == Initializer_weight::Glorot_Uniform){
        weights = matrix_t<T>::Random(N_output,N_input) * std::sqrt(6.0/(N_input + N_output));
    }
    else if constexpr (Iw == Initializer_weight::He_Uniform){
        weights = matrix_t<T>::Random(N_output,N_input) * std::sqrt(6.0/(N_input));
    }
    else if constexpr (Iw == Initializer_weight::Glorot_Norm){
        weights = matrix_t<T>::Ones(N_output,N_input);

        static std::mt19937 gen(42);//std::random_device{}());
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0/(N_input+ N_output)));

        for (int i = 0; i < N_output; i++)
            for (int j = 0; j < N_input; j++)
                weights(i, j) = static_cast<T>(dist(gen));
    }            
    else if constexpr (Iw == Initializer_weight::He_Norm){
        weights = matrix_t<T>::Ones(N_output,N_input);

        static std::mt19937 gen(42); //std::random_device{}());
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0/N_input));

        for (int i = 0; i < N_output; i++)
            for (int j = 0; j < N_input; j++)
                weights(i, j) = static_cast<T>(dist(gen));
    }
    return;
}

template <typename T>
template <Initializer_bias Ib>
inline void Param<T>::initialize_bias(){
    if constexpr (Ib == Initializer_bias::One){
        bias = vector_t<T>::Ones(N_output);
    }
    else if constexpr (Ib == Initializer_bias::Zero){
        bias = vector_t<T>::Zero(N_output);
    }
    else if constexpr (Ib == Initializer_bias::Uniform){
        bias = vector_t<T>::Random(N_output);
    }
    else if constexpr (Ib == Initializer_bias::Normal){
        bias = vector_t<T>::Ones(N_output);

        static std::mt19937 gen(42);//std::random_device{}());
        std::normal_distribution<double> dist(0.0, 0.05);

        for (int i = 0; i < N_output; i++)
            bias(i) = static_cast<T>(dist(gen));
    }
    return;
}

// }

#endif