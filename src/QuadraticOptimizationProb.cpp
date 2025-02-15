#include "../include/QuadraticOptimizationProb.hpp"

double QuadraticOptimizationProb::evaluate(const std::vector<double>& input){
    double x = input[0];

    return (x-1)*(x-1);

}

std::vector<double> QuadraticOptimizationProb::evaluate_gradient(const std::vector<double>& input){
    double x = input[0];
    return {2*(x-1)};
}