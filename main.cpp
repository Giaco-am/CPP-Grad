#include "./include/OptimizationProblem.hpp"
#include "./include/GradientDescent.hpp"
#include "./include/QuadraticOptimizationProb.hpp"
#include "./include/LinearReg.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <random>

int main(){

    // Quadratic example
    std::shared_ptr<OptimizationProblem> problem = std::make_shared<QuadraticOptimizationProb>();
    GradientDescent optimizer(std::move(problem));

    std::vector<double> initial_guess = {10};
    optimizer.set_learning_rate(0.1);
    optimizer.set_max_iters(1000);
    optimizer.set_convergence_threshold(1e-6);

    auto [solution, loss_archive] = optimizer.optimize(initial_guess);

    std::cout << "Solution: " << solution[0] << std::endl;
    std::cout << "Loss: " << loss_archive.back() << std::endl;

    // Generate noisy data for linear regression
    std::vector<double> x_data;
    std::vector<double> y_data;
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> noise(0.0, 0.02); // small noise

        int m = 10; // number of data points
        for(int i = 0; i < m; i++){
            double x = static_cast<double>(i);
            double y = 4.0 + 0.5 * x + noise(gen);
            x_data.push_back(x);
            y_data.push_back(y);
        }
    }

    // Linear regression example
    std::shared_ptr<OptimizationProblem> lin_reg = std::make_shared<LinearReg>(x_data, y_data);
    GradientDescent lin_reg_optimizer(std::move(lin_reg));

    std::vector<double> lin_reg_initial_guess = {0, 0};
    lin_reg_optimizer.set_learning_rate(0.01);
    lin_reg_optimizer.set_max_iters(10000);
    lin_reg_optimizer.set_convergence_threshold(1e-5);

    auto [lin_reg_solution, lin_reg_loss_archive] = lin_reg_optimizer.optimize(lin_reg_initial_guess);

    std::cout << "Linear Regression Solution: " 
              << lin_reg_solution[0] << " " 
              << lin_reg_solution[1] << std::endl;
    std::cout << "Linear Regression Loss: " 
              << lin_reg_loss_archive.back() << std::endl;

    return 0;
}