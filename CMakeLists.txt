cmake_minimum_required(VERSION 3.28)
project(projectPACS LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

add_executable(example main.cpp)
target_link_libraries(example PRIVATE pybind11::embed)