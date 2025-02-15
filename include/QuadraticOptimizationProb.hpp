#ifndef QUADRATICOPTIMIZATIONPROB_HPP
#define QUADRATICOPTIMIZATIONPROB_HPP

#include "OptimizationProblem.hpp"

class QuadraticOptimizationProb: public OptimizationProblem{
    public:
        double evaluate(const std::vector<double>& input) override;

        std::vector<double> evaluate_gradient(const std::vector<double>& input) override;
        
};

#endif