cmake_minimum_required(VERSION 3.12)
project(GradientDescentOptimizer)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Find pybind11 (make sure you have it installed or use it as a submodule)
find_package(pybind11 REQUIRED)

# Build the C++ library (we compile the .cpp files into an object library)
add_library(optimizer 
    src/GradientDescent.cpp
    src/QuadraticOptimizationProb.cpp
    src/LinearReg.cpp
)

# Build the Python module using pybind11.
pybind11_add_module(gdcpp MODULE bindings.cpp)
target_link_libraries(gdcpp PRIVATE optimizer)
