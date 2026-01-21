// #include "Layer.hpp"
// 
// namespace layer{
// 
// class Sigmoid_Layer final: public Layer{
//     public:
//         Sigmoid_Layer(int n_input, int n_output): Layer(n_input, n_output){};
// 
//         Sigmoid_Layer(const Sigmoid_Layer & ) = default;
// 
//         vector & forward_activation() override{
//             for (int i = 0; i < N_output; i++){
//                 output(i) = 1.0 / (1.0 + std::exp(-a[i]));
//             }
// 
//             return output;
//         }
// 
//         std::string layer_type() override{return "Sigmoid Layer";}
// 
// };
// }
// 
// NOTE: Replaced by Dense<T, SigmoidAct> in Dense.hpp for autodiff compatibility.