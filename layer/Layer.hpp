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
        void initialize_weight();

        template<Initializer_bias Ib>
        void initialize_bias();

        vector & forward(const vector & x);

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

