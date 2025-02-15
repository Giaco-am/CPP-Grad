



#include "./include/GradientDescent.hpp"
#include "./include/QuadraticOptimizationProb.hpp"
#include "./include/LinearReg.hpp"

#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
namespace py = pybind11;   
PYBIND11_MODULE(gdcpp, m){

    py::class_<OptimizationProblem, std::shared_ptr<OptimizationProblem>>(m, "OptimizationProblem");


    

    py::class_<GradientDescent, std::shared_ptr<GradientDescent>>(m, "GradientDescent")
        .def(py::init<std::shared_ptr<OptimizationProblem>>())
        .def("set_learning_rate", &GradientDescent::set_learning_rate)
        .def("set_max_iters", &GradientDescent::set_max_iters)
        .def("set_convergence_threshold", &GradientDescent::set_convergence_threshold)
        .def("optimize", &GradientDescent::optimize);

    py::class_<QuadraticOptimizationProb, OptimizationProblem, std::shared_ptr<QuadraticOptimizationProb>>(m, "QuadraticOptimizationProb")
        .def(py::init<>())
        .def("evaluate", &QuadraticOptimizationProb::evaluate)
        .def("evaluate_gradient", &QuadraticOptimizationProb::evaluate_gradient);

    py::class_<LinearReg, OptimizationProblem, std::shared_ptr<LinearReg>>(m, "LinearReg")
        .def(py::init<const std::vector<double>&, const std::vector<double>&>())
        .def("evaluate", &LinearReg::evaluate)
        .def("evaluate_gradient", &LinearReg::evaluate_gradient);   
}