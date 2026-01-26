#ifndef FNN_HPP
#define FNN_HPP

#include <iostream>
#include <vector>
#include <chrono>
#include <autodiff/forward/dual.hpp>
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

template <activation_type A, typename T>
class FNN{
    private:
        const std::vector<int> layer_size;
        std::vector<Dense<T>> layers;  // Dense without Activation template
        typename ActivationSelector<A>::template type <T> activation_function;
        
        // Storage for forward pass (for backpropagation)
        mutable std::vector<vector_t<T>> layer_inputs;  // Stores input to each layer

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
            
            // Reserve space for layer inputs
            layer_inputs.resize(layer_size.size());
        }

        template <Initializer_weight Iw, Initializer_bias Ib>
        void initialize(){
            for (auto & layer : layers){
                layer.template initialize<Iw, Ib>();
            }
        }

        // Forward pass
        vector_t<T> forward(const vector_t<T> & input) const{
            layer_inputs[0] = input;
            vector_t<T> out = input;
            
            // Apply all layers: hidden layers with activation, last layer without
            for(size_t i = 0; i < layers.size() - 1; ++i){
                std::cout << "layer: " << i <<std::endl;
                out = layers[i].forward(out);
                out = activation_function(out);
                layer_inputs[i + 1] = out;
            }
            
            // Last layer: linear only (no activation)
            out = layers.back().forward(out);
            layer_inputs[layers.size()] = out;
            
            return out;
        }

        // Forward pass with a different scalar type than the stored parameters.
        // Useful for autodiff on inputs without turning weights/bias into AD variables.
        template <typename S>
        vector_t<S> forward_mixed(const vector_t<S>& input) const{
            vector_t<S> out = input;

            for(size_t i = 0; i < layers.size() - 1; ++i){
                out = layers[i].forward(out);
                out = activation_function(out);
            }

            out = layers.back().forward(out);
            return out;
        }


        
        // // Compute gradients using autodiff (forward-mode automatic differentiation)
        // std::vector<std::pair<matrix_t<T>, vector_t<T>>> backward(const vector_t<T>& grad_output)
        // {
        //     std::vector<std::pair<matrix_t<T>, vector_t<T>>> gradients(layers.size());
            
        //     // For each weight and bias, compute gradient using autodiff
        //     for(size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx){
        //         int n_out = layers[layer_idx].weights.rows();
        //         int n_in = layers[layer_idx].weights.cols();
                
        //         matrix_t<T> grad_weights(n_out, n_in);
        //         vector_t<T> grad_bias(n_out);
                
        //         // Compute gradient for each weight
        //         for(int i = 0; i < n_out; ++i){
        //             for(int j = 0; j < n_in; ++j){
        //                 using autodiff::dual;
                        
        //                 // Create dual variable for this weight
        //                 dual w_dual = layers[layer_idx].weights(i, j);
        //                 w_dual.val = layers[layer_idx].weights(i, j);
        //                 w_dual.der = 1.0;
                        
        //                 // Forward pass with perturbed weight
        //                 dual out_dual = forward_with_dual_weight(layer_idx, i, j, w_dual);
                        
        //                 // Gradient is: grad_output^T * dout/dw
        //                 grad_weights(i, j) = grad_output(layer_idx, 0) * out_dual.der;
        //             }
        //         }
                
        //         // Compute gradient for each bias
        //         for(int i = 0; i < n_out; ++i){
        //             using autodiff::dual;
                    
        //             dual b_dual = layers[layer_idx].bias(i);
        //             b_dual.val = layers[layer_idx].bias(i);
        //             b_dual.der = 1.0;
                    
        //             dual out_dual = forward_with_dual_bias(layer_idx, i, b_dual);
                    
        //             grad_bias(i) = grad_output(layer_idx, 0) * out_dual.der;
        //         }
                
        //         gradients[layer_idx] = {grad_weights, grad_bias};
        //     }
            
        //     return gradients;
        // }
        
        // // Forward pass with dual weight for gradient computation
        // template <typename DualType>
        // DualType forward_with_dual_weight(size_t layer_idx, int w_row, int w_col, const DualType& w_dual) const
        // {
        //     vector_t<DualType> out = vector_t<DualType>(layer_inputs[layer_idx].size());
        //     for(int i = 0; i < layer_inputs[layer_idx].size(); ++i){
        //         out(i) = DualType(layer_inputs[layer_idx](i), 0.0);
        //     }
            
        //     // Propagate through layers starting from layer_idx
        //     for(size_t i = layer_idx; i < layers.size(); ++i){
        //         vector_t<DualType> z(layers[i].weights.rows());
        //         for(int r = 0; r < layers[i].weights.rows(); ++r){
        //             z(r) = DualType(0.0, 0.0);
        //             for(int c = 0; c < layers[i].weights.cols(); ++c){
        //                 DualType w_val = (i == layer_idx && r == w_row && c == w_col) ? w_dual : DualType(layers[i].weights(r, c), 0.0);
        //                 z(r) = z(r) + w_val * out(c);
        //             }
        //             z(r) = z(r) + DualType(layers[i].bias(r), 0.0);
        //         }
                
        //         if(i < layers.size() - 1){
        //             out = apply_activation_dual(z);
        //         } else {
        //             out = z;  // Last layer: no activation
        //         }
        //     }
            
        //     return out(0);
        // }
        
        // // Forward pass with dual bias for gradient computation
        // template <typename DualType>
        // DualType forward_with_dual_bias(size_t layer_idx, int b_idx, const DualType& b_dual) const
        // {
        //     vector_t<DualType> out = vector_t<DualType>(layer_inputs[layer_idx].size());
        //     for(int i = 0; i < layer_inputs[layer_idx].size(); ++i){
        //         out(i) = DualType(layer_inputs[layer_idx](i), 0.0);
        //     }
            
        //     // Propagate through layers starting from layer_idx
        //     for(size_t i = layer_idx; i < layers.size(); ++i){
        //         vector_t<DualType> z(layers[i].weights.rows());
        //         for(int r = 0; r < layers[i].weights.rows(); ++r){
        //             z(r) = DualType(0.0, 0.0);
        //             for(int c = 0; c < layers[i].weights.cols(); ++c){
        //                 z(r) = z(r) + DualType(layers[i].weights(r, c), 0.0) * out(c);
        //             }
        //             DualType b_val = (i == layer_idx && r == b_idx) ? b_dual : DualType(layers[i].bias(r), 0.0);
        //             z(r) = z(r) + b_val;
        //         }
                
        //         if(i < layers.size() - 1){
        //             out = apply_activation_dual(z);
        //         } else {
        //             out = z;  // Last layer: no activation
        //         }
        //     }
            
        //     return out(0);
        // }
        
        // // Apply activation with dual types (for autodiff)
        // template <typename DualType>
        // vector_t<DualType> apply_activation_dual(const vector_t<DualType>& z) const {
        //     return ActivationSelector<A>::template type<DualType>::template apply<DualType>(z);
        // }
        
        // // Update weights and biases
        // void update_parameters(const std::vector<std::pair<matrix_t<T>, vector_t<T>>>& gradients, T learning_rate){
        //     for(size_t i = 0; i < layers.size(); ++i){
        //         layers[i].weights -= learning_rate * gradients[i].first;
        //         layers[i].bias -= learning_rate * gradients[i].second;
        //     }
        // }

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