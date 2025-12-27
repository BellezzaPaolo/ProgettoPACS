#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include "Param.hpp"

namespace layer{

class Layer: protected Param{
    protected:
        Param grad;

        vector a;
        vector output;
    
    public:
        Layer(int n_input, int n_output):Param(n_input, n_output), grad(n_input, n_output){
            a = vector::Zero(N_output);
            output = vector::Zero(N_output);
        };

        Layer(const Layer & ) = default;

        vector & forward(const vector & x){
            a = weights * x + bias;

            return forward_activation();
        }

        virtual vector & forward_activation() =0;

        virtual std::string layer_type() =0;

        template <Initializer_weight Iw, Initializer_bias Ib>
        void reset(){
            initialize_weight<Iw>();
            initialize_bias<Ib>();
        }

        void print() const{
            std::cout << "weight: " << std::endl;
            std::cout << weights << std::endl;
            std::cout << "bias: " << std::endl;
            std::cout << bias << std::endl;
        }


};

}

#endif