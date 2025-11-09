#include "ParaflowS.hpp"
#include <chrono>

void ParaflowS::optimize(){

    for(size_t iter = 0; iter < max_iterations; iter++){
        double loss_coarse = coarse_solver();
        double loss_fine = fine_solver();

        size_t j = 0;
        correction = parameters - coarse_parameters;
        while(j < n_coarse && stay){
            parameters = coarse_parameters + correction;
            loss_coarse = coarse_solver();

            if (loss_coarse < loss_fine || j == 0) {
                old_correction = parameters;

                loss_fine = loss_coarse;

                cost_history.push_back(loss_coarse);
                parameters_history.push_back(parameters);
                j++;
                
            } 
            else {
                stay = false;
                parameters = old_correction;
            }
        }
        std::cout<<"Iteration "<< iter+1 <<": Cost = "<< loss_fine <<std::endl;

        gradient = gradient_function(parameters);
        if(gradient.norm() < tolerance){
            std::cout << "Converged after " << iter+1 << " iterations." << std::endl;
            break;
        }
    }

    return;
};

double ParaflowS::coarse_solver(){
    gradient = gradient_function(parameters);
    coarse_parameters = parameters - lr_coarse * gradient;
    return loss_function(coarse_parameters);
};

double ParaflowS::fine_solver(){
    double loss = loss_function(parameters);
    cost_history.push_back(loss);

    for(size_t iter=0; iter< n_fine; iter++){
        gradient = gradient_function(parameters);
        parameters = parameters - lr_fine * gradient;
        loss = loss_function(parameters);
        cost_history.push_back(loss);
        parameters_history.push_back(parameters);
    }

    return loss;
};
