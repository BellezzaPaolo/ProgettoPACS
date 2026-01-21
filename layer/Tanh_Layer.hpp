// #include "Layer.hpp"
// 
// namespace layer{
// 
// class Tanh_Layer final: public Layer{
//     public:
//         Tanh_Layer(int n_input, int n_output): Layer(n_input, n_output){};
// 
//         Tanh_Layer(const Tanh_Layer & ) = default;
// 
//         vector & forward_activation() override{
//             for (int i = 0; i < N_output; i++){
//                 output(i) = std::tanh(a[i]);
//             }
//             return output;
// 
//         }
// 
//         std::string layer_type() override{return "Tanh Layer";}
// 
// };
// }
// 
// NOTE: Replaced by Dense<T, TanhAct> in Dense.hpp for autodiff compatibility.