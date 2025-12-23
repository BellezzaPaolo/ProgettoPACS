#include "Layer.hpp"

namespace layer{

class Linear_Layer: public Layer{
    public:
        Linear_Layer(int n_input, int n_output): Layer(n_input, n_output){};

        Linear_Layer(const Linear_Layer & ) = default;

        vector & forward_activation() override{
            output = a;
            return output;
        }

        std::string layer_type() override{return "Linear Layer";}

};
}