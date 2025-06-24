#include <iostream>
#include <pybind11/embed.h>
namespace py = pybind11;

int main(){
    std::cout<<"Hello world"<<std::endl;


    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    py::print("Hello, World!");

    return 0;
}