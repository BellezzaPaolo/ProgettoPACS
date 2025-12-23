#include <iostream>
#include <Eigen/Dense>
#include <random>

#ifndef LAYER_HPP
#define LAYER_HPP

enum class Initializer_weight {One, Uniform, Glorot_Uniform, He_Uniform, Glorot_Norm, He_Norm};
enum class Initializer_bias {One, Zero, Uniform, Normal};

//TODO: the matrix is asumed to be of double, to change visit https://libeigen.gitlab.io/eigen/docs-nightly/group__matrixtypedefs.html#ga2d87d350d9fff1aa3fc27612336d2072
using matrix = Eigen::MatrixXd;
using vector = Eigen::VectorXd;

namespace layer{

class Layer{
    protected:
        const int N_input;
        const int N_output;

        matrix weights;
        vector bias;

        matrix dweights;
        vector dbias;

        vector a;
        vector output;
    
    public:
        Layer(int n_input, int n_output):N_input(n_input), N_output(n_output){
            // initialize the weights and biases in a default mode for robustness
            initialize_weight<Initializer_weight::Glorot_Uniform>();
            initialize_bias<Initializer_bias::Zero>();

            dweights = matrix::Ones(N_output,N_input);
            dbias = vector::Ones(N_output);

            a = vector::Zero(N_output);
            output = vector::Zero(N_output);
        };

        Layer(const Layer & ) = default;

        template<Initializer_weight Iw>
        void initialize_weight(){
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
        void initialize_bias(){
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

        vector & forward(const vector & x){
            a = weights * x + bias;

            return forward_activation();
        }

        virtual vector & forward_activation() =0;

        virtual std::string layer_type() =0;

        void print() const{
            std::cout << "weight: " << std::endl;
            std::cout << weights << std::endl;
            std::cout << "bias: " << std::endl;
            std::cout << bias << std::endl;
        }


};

}
#endif

