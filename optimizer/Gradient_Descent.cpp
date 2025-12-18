#include "Gradient_Descent.hpp"
#include <chrono>

Gradient_Descent::~Gradient_Descent() = default;

void Gradient_Descent::optimize(){
    std::cout<<"Starting optimization with Gradient Descent..."<<std::endl;

    const auto start= std::chrono::steady_clock::now(); //start record the time

    for(size_t iter = 0; iter< max_iterations; iter++){
        gradient = gradient_function(parameters);
        parameters = parameters - lr * gradient;

        double cost = loss_function(parameters);
        cost_history.push_back(cost);
        parameters_history.push_back(parameters);

        std::cout<<"Iteration "<< iter+1 <<": Cost = "<< cost <<", Parameters = [";
        for(int i=0; i<parameters.size(); i++){
            std::cout<< parameters(i);
            if(i < parameters.size()-1) std::cout<< ", ";
        }
        std::cout<<"]"<<std::endl;

        if(gradient.norm() < tolerance){
            const auto end= std::chrono::steady_clock::now(); //end record the time
            std::cout<<"Convergence reached at iteration "<< iter+1 << "training done in " << std::chrono::duration<double>(end-start).count() << " seconds"<<std::endl;
            break;
        }

    }
    return;
}