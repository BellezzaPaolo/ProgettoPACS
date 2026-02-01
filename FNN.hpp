#ifndef FNN_HPP
#define FNN_HPP

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>

/**
 * @file FNN.hpp
 * @brief Fully-connected neural network (feed-forward) implemented with LibTorch.
 *
 * This header provides:
 * - activation and initialization enums
 * - a template network class `FNN<Activation>` that inherits from `torch::nn::Module`
 *
 * The implementation is header-only because it is a class template.
 */

/**
 * @brief Supported activation functions for hidden layers.
 */
enum class Activation { Relu, Tanh, Sigmoid, Linear };

/**
 * @brief Supported weight initialization strategies.
 */
enum class Initializer_weight {One, Uniform, Glorot_Uniform, He_Uniform, Glorot_Norm, He_Norm};

/**
 * @brief Supported bias initialization strategies.
 */
enum class Initializer_bias {Constant, Uniform, Normal};

/**
 * @brief Fully-connected feed-forward neural network.
 *
 * @tparam A Activation used for all hidden layers. The last layer is linear.
 *
 * @details
 * This class builds a sequence of `torch::nn::Linear` layers based on `layer_sizes`.
 * Example: `{2, 50, 50, 1}` builds 3 linear layers: 2→50→50→1.
 *
 * Notes:
 * - The network expects inputs of shape `[N, in_dim]`.
 * - If a 1D tensor `[in_dim]` is provided, it is promoted to `[1, in_dim]`.
 */
template <Activation A>
class FNN final : public torch::nn::Module {
private:
	std::vector<int64_t> layer_sizes_;
	int depth;
	torch::nn::ModuleList layers{nullptr};

	/**
	 * @brief Apply the compile-time activation to a tensor.
	 */
	torch::Tensor apply_activation(const torch::Tensor& x) const{
		if constexpr (A == Activation::Relu)
			return torch::relu(x);
		else if constexpr (A == Activation::Tanh)
			return torch::tanh(x);
		else if constexpr (A == Activation::Sigmoid)
			return torch::sigmoid(x);
		else
			return x; // Linear
	}

	// torch::Tensor apply_activation(const torch::Tensor& x) const
	// {
	// 	switch (activation_){
	// 		case Activation::Relu:    return torch::relu(x);
	// 		case Activation::Tanh:    return torch::tanh(x);
	// 		case Activation::Sigmoid: return torch::sigmoid(x);
	// 		case Activation::Linear:
	// 		default:                 return x;
	// 	}
	// }

public:
	/**
	 * @brief Default constructor.
	 *
	 * @warning The network is not usable until constructed with layer sizes.
	 */
	FNN() = default;

	/**
	 * @brief Construct the network from layer sizes.
	 *
	 * @param layer_sizes Vector of layer widths. Example: `{2, 50, 50, 1}`.
	 *
	 * @throws std::invalid_argument if fewer than 2 sizes are provided.
	 */
	explicit FNN(const std::vector<int64_t>& layer_sizes): layer_sizes_(layer_sizes){
		if(layer_sizes_.size() < 2)
			throw std::invalid_argument("PINN requires at least input and output layer");

		layers = register_module("linear", torch::nn::ModuleList());
		for(size_t i = 1; i < layer_sizes_.size(); ++i){
			layers->push_back(torch::nn::Linear(
				torch::nn::LinearOptions(layer_sizes_[i - 1], layer_sizes_[i])
			));
		}

		depth = layers->size();
	}

	/**
	 * @brief Get the network layer sizes.
	 */
 	const std::vector<int64_t>& get_layer_sizes() const { return layer_sizes_; }

	/**
	 * @brief Get the activation configured for hidden layers.
	 */
	Activation get_activation() const { return A; }

	/**
	 * @brief Forward pass.
	 *
	 * @param x Input tensor of shape `[N, in_dim]` or `[in_dim]`.
	 * @return Output tensor of shape `[N, out_dim]`.
	 */
	torch::Tensor forward(torch::Tensor x){
		// Accept x as [N, in_dim] or [in_dim]. Convert 1D -> [1, in_dim].
		if(x.dim() == 1)
			x = x.unsqueeze(0);

		TORCH_CHECK(!layer_sizes_.empty(), "FNN is not initialized with layer sizes");
		TORCH_CHECK(x.size(-1) == layer_sizes_.front(), "Expected input dim ", layer_sizes_.front(), ", got ", x.size(-1));

		// const int64_t n_layers = static_cast<int64_t>(layers->size());
		
		for(size_t i = 0; i < depth; ++i){
			x = layers[i]->as<torch::nn::Linear>()->forward(x);
			if(i != depth - 1)
				x = apply_activation(x);
		}
		return x;
	}

	/**
	 * @brief Initialize all weights according to the selected strategy.
	 * @tparam Iw Weight initializer.
	 */
	template <Initializer_weight Iw>
	void initialize_weight();

	/**
	 * @brief Initialize all biases according to the selected strategy.
	 * @tparam Ib Bias initializer.
	 * @param constant_value Used only when `Ib == Initializer_bias::Constant`.
	 */
	template <Initializer_bias Ib>
	void initialize_bias(const double constant_value = 0.0);

	/**
	 * @brief Initialize both weights and biases.
	 * @tparam Iw Weight initializer.
	 * @tparam Ib Bias initializer.
	 * @param bias_constant_value Used only when `Ib == Initializer_bias::Constant`.
	 */
	template <Initializer_weight Iw, Initializer_bias Ib>
	void initialize(const double bias_constant_value = 0.0){
		initialize_weight<Iw>();
		initialize_bias<Ib>(bias_constant_value);

		return;
	}

	/**
	 * @brief Print a human-readable summary of the network architecture.
	 */
	void print() const{
		std::cout << "The network has shape: [ ";
		for(size_t j=0; j<layer_sizes_.size(); ++j){
			std::cout << layer_sizes_[j];
			if (j + 1 < layer_sizes_.size()) std::cout << ", ";
		}
		std::cout << " ]" << std::endl;

		std::cout << "Layers: " << std::endl;
		for (size_t i = 1; i < layer_sizes_.size(); ++i){
			std::string act_type;
			if(i < layer_sizes_.size() - 1){
				if constexpr (A == Activation::Relu) act_type = "relu";
				else if constexpr (A == Activation::Tanh) act_type = "tanh";
				else if constexpr (A == Activation::Sigmoid) act_type = "sigmoid";
				else act_type = "linear";
			} 
			else{
				act_type =  "linear/output";
			}
			std::cout << "Dense " << (i) << " " << act_type << " (" 
						<< layer_sizes_[i-1] << "x" << layer_sizes_[i] << ")" << std::endl;
		}
		std::cout << std::endl;
	}
};

template <Activation A>
template <Initializer_weight Iw>
void FNN<A>::initialize_weight(){
	for(auto& m : modules(/*include_self=*/false)){
		if(auto* lin = dynamic_cast<torch::nn::LinearImpl*>(m.get())){
			const auto N_out = lin->weight.size(0);
			const auto N_in  = lin->weight.size(1);

			if constexpr (Iw == Initializer_weight::One){
				torch::nn::init::ones_(lin->weight);
			}
			else if constexpr (Iw == Initializer_weight::Uniform){
				// Uniform in [-1, 1]
				torch::nn::init::uniform_(lin->weight, -1.0, 1.0);
			}
			else if constexpr (Iw == Initializer_weight::Glorot_Uniform){
				double limit = std::sqrt(6.0 / static_cast<double>(N_in + N_out));
				torch::nn::init::uniform_(lin->weight, -limit, limit);
			}
			else if constexpr (Iw == Initializer_weight::He_Uniform){
				double limit = std::sqrt(6.0 / static_cast<double>(N_in));
				torch::nn::init::uniform_(lin->weight, -limit, limit);
			}
			else if constexpr (Iw == Initializer_weight::Glorot_Norm){
				double stddev = std::sqrt(2.0 / static_cast<double>(N_in + N_out));
				torch::nn::init::normal_(lin->weight, 0.0, stddev);
			}
			else if constexpr (Iw == Initializer_weight::He_Norm){
				double stddev = std::sqrt(2.0 / static_cast<double>(N_in));
				torch::nn::init::normal_(lin->weight, 0.0, stddev);
			}
		}
	}
}

template <Activation A>
template <Initializer_bias Ib>
void FNN<A>::initialize_bias(const double constant_value){
	for(auto& m : modules(/*include_self=*/false)){
		if(auto* lin = dynamic_cast<torch::nn::LinearImpl*>(m.get())){
			if constexpr (Ib == Initializer_bias::Constant){
				torch::nn::init::constant_(lin->bias, constant_value);
			}
			else if constexpr (Ib == Initializer_bias::Uniform){
				// Uniform in [-1, 1]
				torch::nn::init::uniform_(lin->bias, -1.0, 1.0);
			}
			else if constexpr (Ib == Initializer_bias::Normal){
				// Normal with small std-dev
				torch::nn::init::normal_(lin->bias, 0.0, 0.05);
			}
		}
	}
}

#endif