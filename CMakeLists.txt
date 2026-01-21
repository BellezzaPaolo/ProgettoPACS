cmake_minimum_required(VERSION 3.28)
project(projectPACS LANGUAGES CXX)

# Use C++17 or later
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # For pybind11 safety


# --- pybind11 from Conda ---
set(pybind11_DIR "/home/paolo/miniforge3/envs/pacs/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# --- Include directories ---
include_directories(${.})

# Eigen (system)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# --- Source files ---
set(SOURCES
    main.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/layer/Param.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/Gradient_Descent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/ParaflowS.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/boundary_condition/Boundary_Condition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/boundary_condition/Dirichlet_BC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Pde.cpp
)

# --- Create executable ---
add_executable(example ${SOURCES})

# --- Link pybind11 ---
target_link_libraries(example PRIVATE pybind11::embed Eigen3::Eigen)

# --- Set environment variables for running the executable ---
set(PYTHON_ENV_HOME "/home/paolo/miniforge3/envs/pacs")
set(PYTHON_LIB_PATH "${PYTHON_ENV_HOME}/lib")

# Add a custom target to run example with proper environment
add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E env 
        PYTHONHOME=${PYTHON_ENV_HOME}
        LD_LIBRARY_PATH=${PYTHON_LIB_PATH}:$ENV{LD_LIBRARY_PATH}
        $<TARGET_FILE:example>
    DEPENDS example
    COMMENT "Running example with Python environment..."
)