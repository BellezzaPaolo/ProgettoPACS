#include "Layer.hpp"

namespace layer{

    template<Initializer_weight Iw>
    void Layer::initialize_weight(){
        if constexpr (Iw == Initializer_weight::One){
            weights = matrix::Ones(N_output,N_input);
        }
        else if constexpr (Iw == Initializer_weight::Uniform){
            weights = matrix::Random(N_output,N_input);
        }
        else if constexpr (Iw == Initializer_weight::Glorot_Uniform){
            weights = matrix::Random(N_output,N_input) * std::sqrt(6.0/(N_input + N_output));
        }
        else if constexpr (Iw == Initializer_weight::He_Uniform){
            weights = matrix::Random(N_output,N_input) * std::sqrt(6.0/(N_input));
        }
        else if constexpr (Iw == Initializer_weight::Glorot_Norm){
            weights = matrix::Ones(N_output,N_input);

            static std::mt19937 gen(42);//std::random_device{}());
            std::normal_distribution<double> dist(0.0, std::sqrt(2.0/(N_input+ N_output)));

            for (int i = 0; i < N_output; i++)
                for (int j = 0; j < N_input; j++)
                    weights(i, j) = dist(gen);
        }            
        else if constexpr (Iw == Initializer_weight::He_Norm){
            weights = matrix::Ones(N_output,N_input);

            static std::mt19937 gen(42); //std::random_device{}());
            std::normal_distribution<double> dist(0.0, std::sqrt(2.0/N_input));

            for (int i = 0; i < N_output; i++)
                for (int j = 0; j < N_input; j++)
                    weights(i, j) = dist(gen);
        }
        return;
    }

    template<Initializer_bias Ib>
    void Layer::initialize_bias(){
        if constexpr (Ib == Initializer_bias::One){
            bias = vector::Ones(N_output);
        }
        else if constexpr (Ib == Initializer_bias::Zero){
            bias = vector::Zero(N_output);
        }
        else if constexpr (Ib == Initializer_bias::Uniform){
            bias = vector::Random(N_output);
        }
        else if constexpr (Ib == Initializer_bias::Normal){
            bias = vector::Ones(N_output);

            static std::mt19937 gen(42);//std::random_device{}());
            std::normal_distribution<double> dist(0.0, 0.05);

            for (int i = 0; i < N_output; i++)
                bias(i) = dist(gen);
        }
        return;
    }

    vector & Layer::forward(const vector & x){
        a = weights * x + bias;

        return forward_activation();
    }

    // Explicit template instantiations for all used combinations
    template void Layer::initialize_weight<Initializer_weight::One>();
    template void Layer::initialize_weight<Initializer_weight::Uniform>();
    template void Layer::initialize_weight<Initializer_weight::Glorot_Uniform>();
    template void Layer::initialize_weight<Initializer_weight::He_Uniform>();
    template void Layer::initialize_weight<Initializer_weight::Glorot_Norm>();
    template void Layer::initialize_weight<Initializer_weight::He_Norm>();

    template void Layer::initialize_bias<Initializer_bias::Zero>();
    template void Layer::initialize_bias<Initializer_bias::One>();
    template void Layer::initialize_bias<Initializer_bias::Uniform>();
    template void Layer::initialize_bias<Initializer_bias::Normal>();

}