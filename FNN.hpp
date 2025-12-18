#ifndef FNN_HPP
#define FNN_HPP

#include <iostream>
#include <vector>
#include <chrono>
#include "layer/Layer.hpp"

enum class activation_type {relu, linear, tanh, sigmoid};

template <activation_type A>
class FNN{
    private:
        const std::vector<int> layer_size;
        int depth = 0;
        std::vector<Layer> params;
        vector output;

        // void activation_function(vector & x){
        //     if constexpr (A == activation_type::relu){
        //         for (int i = 0; i < x.size(); i++){
        //             x(i) = std::max(0.0, x(i));
        //         }
        //     }
        //     else if constexpr (A == activation_type::linear){
        //         // do nothing
        //     }
        //     else if constexpr (A == activation_type::tanh){
        //         for (int i = 0; i < x.size(); i++){
        //             x(i) = std::tanh(x(i));
        //         }
        //     }
        //     else if constexpr (A == activation_type::sigmoid){
        //         for (int i = 0; i < x.size(); i++){
        //             x(i) = 1.0 / (1.0 + std::exp(-x(i)));
        //         }
        //     }
        //     return;
        // }

    public:
        FNN(const std::vector<int> & Layer_Size):layer_size(Layer_Size){
            if (layer_size.empty()){
                std::cerr << "The network must have at least one layer." << std::endl;
                throw std::invalid_argument("Invalid layer size");
            }

            if (layer_size.size() == 1){
                std::cerr << "The network must have at least an input layer and an output layer." << std::endl;
                throw std::invalid_argument("Invalid layer size");
            }

            for (size_t i = 1; i < layer_size.size(); i++){
                add_layer<A>(layer_size[i-1], layer_size[i]);
            }

            output = vector::Zero(layer_size.back());
        }

        template <activation_type A>
        void add_layer(int n_input, int n_output){
            if constexpr (A == relu){
                Relu_Layer new_layer(n_input, n_output);
            }
            else if constexpr (A == linear){
                Linear_layer new_layer(n_input, n_output);
            }
            else if constexpr( A == tanh){
                Tanh_Layer new_layer(n_input, n_output);
            }
            else if constexpr (A == sigmoid){
                Sigmoid_Layer new_layer(n_input, n_output);                
            }
            params.push_back(new_layer);
            depth += 1;
            return;
        }

        template <Initializer_weight Iw, Initializer_bias Ib>
        void initialize(){
            for (auto& layer : params){
                layer.initialize_weight<Iw>();
                layer.initialize_bias<Ib>();
            }
            return;
        }

        vector forward(const vector & input){
            output = input;

            for(size_t i = 0; i < params.size() - 1; i++){
                output = params[i].forward(output);
                activation_function(output);
            }

            output = params.back().forward(output);

            return output;
        }


        void print() const{
            int i = 1;
            for (auto& layer : params){
                std::cout << "Layer "<< i <<":"<< std::endl;
                layer.print();
                i ++;
            }
            return;
        }
};
#endif