#ifndef FNN_HPP
#define FNN_HPP

#include <iostream>
#include <vector>
#include <chrono>
#include "layer/Dense.hpp"
#include "layer/activation.hpp"

enum class activation_type {relu, linear, tanh, sigmoid};

// Map enum to activation functor
template <activation_type A>
struct ActivationSelector;

template <>
struct ActivationSelector<activation_type::relu> { template <typename T> using type = ReluAct; };
template <>
struct ActivationSelector<activation_type::linear> { template <typename T> using type = LinearAct; };
template <>
struct ActivationSelector<activation_type::tanh> { template <typename T> using type = TanhAct; };
template <>
struct ActivationSelector<activation_type::sigmoid> { template <typename T> using type = SigmoidAct; };

template <activation_type A, typename T = double>
class FNN{
    private:
        const std::vector<int> layer_size;
        std::vector<Dense<T, typename ActivationSelector<A>::template type<T>>> layers;

    public:
        FNN(const std::vector<int> & Layer_Size):layer_size(Layer_Size){
            if (layer_size.size() < 2){
                std::cerr << "The network must have at least an input and an output layer." << std::endl;
                throw std::invalid_argument("Invalid layer size");
            }

            // Create all layers uniformly
            for (size_t i = 1; i < layer_size.size(); ++i){
                layers.emplace_back(layer_size[i-1], layer_size[i]);
            }
        }

        template <Initializer_weight Iw, Initializer_bias Ib>
        void initialize(){
            for (auto & layer : layers){
                layer.template initialize<Iw, Ib>();
            }
        }

        vector_t<T> forward(const vector_t<T> & input) const{
            vector_t<T> out = input;
            // Apply activation for all layers except the last
            for(size_t i = 0; i < layers.size() - 1; ++i){
                out = layers[i](out);
            }
            // Last layer: apply only linear transformation (Dense but skip activation)
            out = layers.back().linear(out);
            return out;
        }

        void print(bool print_weights = false) const{
            std::cout << "The network has shape: [ ";
            for(size_t j=0; j<layer_size.size(); ++j){
                std::cout << layer_size[j];
                if (j + 1 < layer_size.size()) std::cout << ", ";
            }
            std::cout << " ]" << std::endl;

            std::cout << "Layers: " << std::endl;
            for (size_t i = 0; i < layers.size(); ++i){
                std::string act_type;
                if(i < layers.size() - 1){
                    if constexpr (A == activation_type::relu) act_type = "relu";
                    else if constexpr (A == activation_type::tanh) act_type = "tanh";
                    else if constexpr (A == activation_type::sigmoid) act_type = "sigmoid";
                    else act_type = "linear";
                } 
                else{
                    act_type =  "linear/output";
                }
                std::cout << "Dense " << (i+1) << " " << act_type << " (" 
                          << layer_size[i] << "x" << layer_size[i+1] << ")" << std::endl;
                if (print_weights){
                    layers[i].print();
                }
            }
            std::cout << std::endl;
        }
};
#endif