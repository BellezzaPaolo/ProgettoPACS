#ifndef FNN_HPP
#define FNN_HPP

#include <iostream>
#include <vector>
#include <chrono>
#include "layer/Linear_Layer.hpp"
#include "layer/Relu_Layer.hpp"
#include "layer/Sigmoid_Layer.hpp"
#include "layer/Tanh_Layer.hpp"

enum class activation_type {relu, linear, tanh, sigmoid};

template <activation_type A>
class FNN{
    private:
        const std::vector<int> layer_size;
        int depth = 0;
        std::vector<layer::Layer*> params;
        vector output;

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

            for (size_t i = 1; i < layer_size.size() - 1; i++){
                add_layer(layer_size[i-1], layer_size[i]);
            }

            params.push_back(new layer::Linear_Layer(*(layer_size.end()-2),*(layer_size.end()-1)));

            output = vector::Zero(layer_size.back());
        }

        void add_layer(int n_input, int n_output){
            if constexpr (A == activation_type::relu){
                params.push_back(new layer::Relu_Layer(n_input, n_output));
            }
            else if constexpr (A == activation_type::linear){
                params.push_back(new layer::Linear_Layer(n_input, n_output));
            }
            else if constexpr( A == activation_type::tanh){
                params.push_back(new layer::Tanh_Layer(n_input, n_output));
            }
            else if constexpr (A == activation_type::sigmoid){
                params.push_back(new layer::Sigmoid_Layer(n_input, n_output));
            }

            depth += 1;
            return;
        }

        template <Initializer_weight Iw, Initializer_bias Ib>
        void initialize(){
            for (auto layer : params){
                layer->initialize_weight<Iw>();
                layer->initialize_bias<Ib>();
            }
            return;
        }

        vector & forward(const vector & input){
            output = input;

            for(size_t i = 0; i < params.size(); i++){
                output = params[i]->forward(output);
            }

            return output;
        }


        void print(bool print = false) const{
            int i = 1;

            std::cout << "The network has shape: " << std::endl;
            std::cout <<"[ ";
            for(size_t j=0; j<layer_size.size();++j){
                std::cout << layer_size[j];
                if (j + 1 < layer_size.size()){
                    std::cout << ", ";
                }
            }
            std::cout << " ]"<<std::endl;

            std::cout << "Formed by: " << std::endl;
            std::cout<<std::endl;

            for (auto layer : params){
                std::cout <<layer->layer_type()<<" "<< i <<":"<< std::endl;
                if (print){
                    layer->print();
                }
                else{
                    std::cout<<"    |"<< std::endl;
                    std::cout<<"    V"<< std::endl;
                }
                i ++;
            }
            std::cout<<std::endl;
            std::cout<<"End of network"<<std::endl;
            return;
        }
};
#endif