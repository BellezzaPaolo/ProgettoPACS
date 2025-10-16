#include <iostream>
#include <vector>
#include <functional>
#include <pybind11/embed.h>

namespace py = pybind11;

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

class Optimizer{
    private:
        std::vector<double> parameters;
        std::function<double(std::vector<double>&)> loss_function;
        std::function<std::vector<double>(std::vector<double>&)> gradient_function;
        int max_iterations;
        double tolerance;
        std::vector<double> cost_history;
        std::vector<std::vector<double>> parameters_history;

    public:
        Optimizer(std::function<double(std::vector<double>&)> f, std::function<std::vector<double>(std::vector<double>&)> df, int max_iter, double tol,int dim): 
        parameters(dim, 0.0),loss_function(f), gradient_function(df), max_iterations(max_iter),tolerance(tol){
            cost_history.push_back(loss_function(parameters));
            for (size_t i=0; i<cost_history.size(); i++) {
                std::cout << "Cost at iteration " << i << ": " << cost_history[i] << std::endl;
            }
        };

        int get_max_iterations() const {return max_iterations;}
        void set_max_iterations(int max_iter) {max_iterations = max_iter;}

        double get_tolerance() const {return tolerance;}
        void set_tolerance(double tol) {tolerance = tol;}

        std::vector<double> get_parameters() const {return parameters;}
        void set_parameters(const std::vector<double>& params) {parameters = params;}

        void plot_cost_history() const {
            py::module_ plt = py::module_::import("matplotlib.pyplot");

            // TODO: make work the direct cast from std::vector to py::list
            // py::list py_cost = py::cast(cost_history);
            py::list py_cost(cost_history.size());
            for(size_t i = 0; i<cost_history.size();i++){
                py_cost[i]  = cost_history[i];
            }
            plt.attr("plot")(py_cost);

            plt.attr("xlabel")("Iteration");
            plt.attr("ylabel")("Cost");
            plt.attr("title")("Cost History");
            plt.attr("show")();
        }

        virtual void optimize() =0;

};
#endif