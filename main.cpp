#include <iostream>
#include <pybind11/embed.h>
namespace py = pybind11;

int main(){
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    py::module_ sys = py::module_::import("sys");
    // py::print(sys.attr("path"));
    sys.attr("path").attr("append")("/home/paolo/Desktop/ProgettoPACS"); // add to the working directory to reach pde.py
    // py::print(sys.attr("path"));
    py::module_ dde = py::module_::import("deepxde"); //import deepxde
    py::module_ pde = py::module_::import("pde"); // import pde

    // py::print("Hello, World!");
    // py::object A = np.attr("mean")(
    //     py::make_tuple(1., 2., 3.)
    // );
    // double a = A.cast<double>();
    // std::cout<<a<<std::endl;


    //initialize the geometry
    py::object geom = dde.attr("geometry").attr("Interval")(-1, 1);
    py::object bc_l = dde.attr("icbc").attr("DirichletBC")(geom, pde.attr("func"), pde.attr("boundary_l"));

    py::print(bc_l);

    return 0;
}