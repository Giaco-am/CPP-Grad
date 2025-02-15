#ifndef GRADIENTDESCENT_HPP
#define GRADIENTDESCENT_HPP


#include "OptimizationProblem.hpp"

#include <vector>
#include <memory>
#include <utility>
#include <stdexcept>

class GradientDescent{
    public: 

        GradientDescent(std::shared_ptr<OptimizationProblem> problem);

        void set_learning_rate(double lr);

        void set_max_iters(int max_iters);

        void set_convergence_threshold(double threshold);

        std::pair<std::vector<double>, std::vector<double>> optimize(const std::vector<double>& initial_guess);



        private: 
            std::shared_ptr<OptimizationProblem> problem_;
            double learning_rate_;
            int max_iters_;
            double convergence_threshold_;

};

#endif