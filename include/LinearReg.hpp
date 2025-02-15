#ifndef LINEARREG_HPP
#define LINEARREG_HPP

#include "OptimizationProblem.hpp"
#include <vector>


class LinearReg: public OptimizationProblem{
    public:
        LinearReg(const std::vector<double>& x_data, const std::vector<double>& y_data);

        double evaluate(const std::vector<double>& theta) override;

        std::vector<double> evaluate_gradient(const std::vector<double>& theta) override;

    private:
        std::vector<double> x_data_;
        std::vector<double> y_data_;
        

};





#endif