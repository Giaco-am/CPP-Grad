#include "../include/GradientDescent.hpp"
#include <cmath>
#include <memory>

GradientDescent::GradientDescent(std::shared_ptr<OptimizationProblem> problem): problem_(std::move(problem)), learning_rate_(0.01), max_iters_(500), convergence_threshold_(1e-6) {}


void GradientDescent::set_learning_rate(double lr){
    learning_rate_ = lr;
}

void GradientDescent::set_max_iters(int max_iters){
    max_iters_ = max_iters;
}

void GradientDescent::set_convergence_threshold(double threshold){
    convergence_threshold_ = threshold;
}

std::pair<std::vector<double>, std::vector<double>> GradientDescent::optimize(const std::vector<double>& initial_guess){

    std::vector<double> current_guess = initial_guess;
    std::vector<double> loss_archive;
    double loss = problem_->evaluate(current_guess);
    loss_archive.push_back(loss);

    for(int i=0; i<max_iters_; i++){

        std::vector<double> grad = problem_->evaluate_gradient(current_guess);
        std::vector<double> next(current_guess.size());

        for (size_t j=0; j< current_guess.size(); j++){
            next[j] = current_guess[j] - learning_rate_ * grad[j];
        }

        double next_loss = problem_->evaluate(next);
        loss_archive.push_back(next_loss);

        if (std::fabs(next_loss - loss) < convergence_threshold_){
            return {current_guess, loss_archive};
        }
        
        current_guess = next;
        loss = next_loss;
    }
    throw std::runtime_error("Gradient Descent did not converge");
} 