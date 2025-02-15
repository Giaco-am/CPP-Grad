#include "../include/LinearReg.hpp"
#include <cstddef>

LinearReg::LinearReg(const std::vector<double>& x_data, const std::vector<double>& y_data): x_data_(x_data), y_data_(y_data){};

double LinearReg::evaluate(const std::vector<double>& theta){
    double loss = 0;
    size_t m = x_data_.size();
    for (size_t i=0; i<m; i++){
        double pred = theta[0] + theta[1] * x_data_[i];
        double error = pred - y_data_[i];
        loss += error * error;
    }
    return loss / (2*m);
}


std::vector<double> LinearReg::evaluate_gradient(const std::vector<double>& theta){
    double grad0 = 0;
    double grad1 = 0;
    size_t m = x_data_.size();

    for(size_t i=0; i<m; i++){
        double pred = theta[0] + theta[1] * x_data_[i];
        double error = pred - y_data_[i];
        grad0 += error;
        grad1 += error * x_data_[i];
    }
    grad0 /= m;
    grad1 /= m;
    return {grad0, grad1};
}