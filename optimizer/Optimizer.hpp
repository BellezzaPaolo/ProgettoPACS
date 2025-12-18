#include <iostream>
#include <vector>
#include <functional>
#include <pybind11/embed.h>
#include <Eigen/Dense>

namespace py = pybind11;

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

class Optimizer{
    protected:
        Eigen::Matrix<double,2,1> parameters;
        Eigen::Matrix<double,2,1> gradient;
        std::function<double(Eigen::Matrix<double,2,1>&)> loss_function;
        std::function<Eigen::Matrix<double,2,1>(Eigen::Matrix<double,2,1>&)> gradient_function;
        int max_iterations;
        double tolerance;
        std::vector<double> cost_history;
        std::vector<Eigen::Matrix<double,2,1>> parameters_history;

    public:
        Optimizer(std::function<double(Eigen::Matrix<double,2,1>&)> f, std::function<Eigen::Matrix<double,2,1>(Eigen::Matrix<double,2,1>&)> df, int max_iter, double tol): 
        parameters(), gradient(), loss_function(f), gradient_function(df), max_iterations(max_iter),tolerance(tol)
        {
            // proper initialization for fixed-size Eigen vectors
            parameters << 2.0, 2.0;
            gradient << 1.0, 1.0;
            cost_history.push_back(loss_function(parameters));
        };

        inline int get_max_iterations() const {return max_iterations;}
        inline void set_max_iterations(int max_iter) {max_iterations = max_iter;}

        inline double get_tolerance() const {return tolerance;}
        inline void set_tolerance(double tol) {tolerance = tol;}

        inline Eigen::Matrix<double,2,1> get_parameters() const {return parameters;}
        inline void set_parameters(const Eigen::Matrix<double,2,1>& params) {parameters = params;}
        void print_parameters() const {
            std::cout << "parameters size: " << parameters.size() << std::endl;
            for (int i = 0; i < parameters.size(); ++i) {
                std::cout << parameters(i) << " ";
            }
            std::cout << std::endl;
        }

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

        void plot_parameters_history() const {
            py::module_ plt = py::module_::import("matplotlib.pyplot");
            py::module_ np = py::module_::import("numpy");

            py::list x_vals(parameters_history.size());
            py::list y_vals(parameters_history.size());

            double minX = 100, maxX = -100, minY = 100, maxY = -100;

            for(size_t i = 0; i<parameters_history.size();i++){
                x_vals[i]  = parameters_history[i](0);
                y_vals[i]  = parameters_history[i](1);
                if(parameters_history[i](0) < minX) minX = parameters_history[i](0);
                if(parameters_history[i](0) > maxX) maxX = parameters_history[i](0);
                if(parameters_history[i](1) < minY) minY = parameters_history[i](1);
                if(parameters_history[i](1) > maxY) maxY = parameters_history[i](1);
            }
            // keep x_vals/y_vals as Python lists (they are py::list) and convert C++ vectors to python objects below

            double hx = (maxX - minX) * 0.01;
            double hy = (maxY - minY) * 0.01;

            // create 2D grids (meshgrid) and evaluate loss on them (Z will be 2D)
            const size_t Nx = 100;
            const size_t Ny = 100;
            std::vector<double> X(Nx), Y(Ny);
            for (size_t j = 0; j < Nx; ++j) {
                X[j] = minX + hx * static_cast<double>(j);
            }
            for (size_t i = 0; i < Ny; ++i) {
                Y[i] = minY + hy * static_cast<double>(i);
            }

            // convert X and Y to Python lists and create meshgrid in numpy
            py::list Xpy(Nx), Ypy(Ny);
            for (size_t j = 0; j < Nx; ++j) Xpy[j] = X[j];
            for (size_t i = 0; i < Ny; ++i) Ypy[i] = Y[i];

            py::object mesh = np.attr("meshgrid")(Xpy, Ypy);
            py::object Xnp = mesh.attr("__getitem__")(0);
            py::object Ynp = mesh.attr("__getitem__")(1);

            // build Z as a list of rows (Ny x Nx)
            py::list Zpy(Ny);
            for (size_t i = 0; i < Ny; ++i) {
                py::list row(Nx);
                for (size_t j = 0; j < Nx; ++j) {
                    Eigen::Matrix<double,2,1> pos;
                    pos << X[j], Y[i];
                    row[j] = loss_function(pos);
                }
                Zpy[i] = row;
            }

            py::object Znp = np.attr("array")(Zpy);
            
            plt.attr("figure")();
            plt.attr("contourf")(Xnp, Ynp, Znp, 50, py::arg("cmap")="viridis");
            plt.attr("plot")(x_vals, y_vals, "o-");
            plt.attr("xlabel")("Parameter 1");
            plt.attr("ylabel")("Parameter 2");
            plt.attr("title")("Parameters History");
            plt.attr("show")();
        }
        virtual void optimize() =0;

};
#endif