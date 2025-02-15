#ifndef OPTIMIZATIONPROBLEM_HPP
#define OPTIMIZATIONPROBLEM_HPP

#include <vector>

class OptimizationProblem {

    public:
        virtual ~OptimizationProblem() = default;


        virtual double evaluate(const std::vector<double>& input) = 0;

        virtual std::vector<double> evaluate_gradient(const std::vector <double>& input) = 0;

};

#endif