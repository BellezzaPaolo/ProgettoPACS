cmake_minimum_required(VERSION 3.28)
project(projectPACS LANGUAGES CXX)

# Use C++17 or later
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # For pybind11 safety
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# --- pybind11 from Conda ---
set(pybind11_DIR "/home/paolo/miniforge3/envs/pacs/share/cmake/pybind11")

# --- libtorch ---
# Point this to the folder where you extracted libtorch
list(APPEND CMAKE_PREFIX_PATH "/home/paolo/libtorch")
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)


# --- Include directories ---
include_directories(${.})

# --- Common source files (shared by all executables) ---
set(COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/boundary_condition/Boundary_Condition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/boundary_condition/Dirichlet_BC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pde/Pde.cpp
)

function(add_pacs_executable target_name entry_source)
    add_executable(${target_name}
        ${entry_source}
        ${COMMON_SOURCES}
    )

    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
    )

    target_link_libraries(${target_name} PRIVATE
        pybind11::embed
        ${TORCH_LIBRARIES}
    )
endfunction()

# --- Executables ---
add_pacs_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
add_pacs_executable(poisson_lshape ${CMAKE_CURRENT_SOURCE_DIR}/PoissonLshape.cpp)

# Torch provides recommended compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# --- Set environment variables for running the executable ---
set(PYTHON_ENV_HOME "/home/paolo/miniforge3/envs/pacs")
set(PYTHON_LIB_PATH "${PYTHON_ENV_HOME}/lib")

function(add_pacs_run_target run_target_name exe_target_name)
    add_custom_target(${run_target_name}
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONHOME=${PYTHON_ENV_HOME}
            LD_LIBRARY_PATH=${PYTHON_LIB_PATH}:$ENV{LD_LIBRARY_PATH}
            $<TARGET_FILE:${exe_target_name}>
        DEPENDS ${exe_target_name}
        COMMENT "Running ${exe_target_name} with Python environment..."
    )
endfunction()

# Run targets
add_pacs_run_target(run_test test)
add_pacs_run_target(run_poissonlshape poisson_lshape)