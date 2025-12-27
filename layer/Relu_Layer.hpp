#include "Layer.hpp"

namespace layer{

class Relu_Layer final: public Layer{
    public:
        Relu_Layer(int n_input, int n_output): Layer(n_input, n_output){};

        Relu_Layer(const Relu_Layer & ) = default;

        vector & forward_activation() override{

            for (size_t i = 0; i < N_output; i++){
                output(i) = std::max(0.0, a[i]);
            }

            return output;
        };

        std::string layer_type() override{return "Relu Layer";}

};
}