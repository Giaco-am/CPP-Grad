import numpy as np
import matplotlib.pyplot as plt
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, "build"))
from gdcpp import QuadraticOptimizationProb, LinearReg, GradientDescent


output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)
# -----------------------------
# Problem 1: Quadratic Function
# -----------------------------
quad_problem = QuadraticOptimizationProb()
gd_quad = GradientDescent(quad_problem)
gd_quad.set_learning_rate(0.1)
gd_quad.set_max_iters(100)
gd_quad.set_convergence_threshold(1e-6)
initial_guess = [0.0]  # starting point for x

solution, cost_history = gd_quad.optimize(initial_guess)
print("Problem 1: Quadratic Optimization")
print("Solution: ", solution)
print("Exact solution is x = 1, error =", abs(solution[0] - 1))

plt.figure()
plt.plot(cost_history, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs. Iterations (Quadratic Function)")
plt.grid(True)
# Replace the first plt.show() with:
plt.savefig(os.path.join(output_dir, "quadratic_optimization.png"))
plt.close()

# --------------------------------
# Problem 2: Linear Regression
# --------------------------------
# Generate synthetic data: y = 4 + 0.5*x + noise
np.random.seed(0)
x_data = np.linspace(0, 10, 50)
noise = np.random.randn(50) * 0.5
y_data = 4 + 0.5 * x_data + noise

# Create the problem and optimizer.
linreg_problem = LinearReg(x_data.tolist(), y_data.tolist())
gd_linreg = GradientDescent(linreg_problem)
gd_linreg.set_learning_rate(0.01)
gd_linreg.set_max_iters(10000)
gd_linreg.set_convergence_threshold(1e-8)
initial_guess = [0.0, 0.0]  # starting guess for theta0 and theta1

solution_linreg, cost_history_linreg = gd_linreg.optimize(initial_guess)
print("\nProblem 2: Linear Regression")
print("Optimized parameters: theta0 = {:.4f}, theta1 = {:.4f}".format(solution_linreg[0], solution_linreg[1]))







# Plot the fitted line along with data.
plt.figure()
plt.scatter(x_data, y_data, label='Data')
predictions = [solution_linreg[0] + solution_linreg[1]*x for x in x_data]
plt.plot(x_data, predictions, color='red', label='Fitted Line')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
# Replace the second plt.show() with:
plt.savefig(os.path.join(output_dir, "linear_regression_fit.png"))
plt.close()

# Plot cost function history.
plt.figure()
plt.plot(cost_history_linreg, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs. Iterations (Linear Regression)")
plt.grid(True)
# Replace the third plt.show() with:
plt.savefig(os.path.join(output_dir, "linear_regression_cost.png"))
plt.close()

# Evaluate final MSE cost function.
final_cost = linreg_problem.evaluate(solution_linreg)
print("Final MSE Cost:", final_cost)



# --------------------------------
# Validation using SciPy functions
# --------------------------------
from scipy.optimize import minimize_scalar
from scipy.stats import linregress

# Validate Problem 1 (Quadratic Function)
# Assuming the quadratic function is f(x) = (x - 1)**2
res = minimize_scalar(lambda x: (x - 1)**2)
print("\nSciPy minimize_scalar result for Quadratic Function:")
print("Solution: x =", res.x, ", f(x) =", res.fun)
print("Error from exact solution (1):", abs(res.x - 1))

# Validate Problem 2 (Linear Regression)
lr_result = linregress(x_data, y_data)
print("\nSciPy linregress result for Linear Regression:")
print("Intercept =", lr_result.intercept, ", Slope =", lr_result.slope)

print("\nComparison with GradientDescent results:")
print("GradientDescent optimized parameters: theta0 = {:.4f}, theta1 = {:.4f}".format(solution_linreg[0], solution_linreg[1]))
print("Difference in intercepts: {:.4f}".format(abs(solution_linreg[0] - lr_result.intercept)))
print("Difference in slopes: {:.4f}".format(abs(solution_linreg[1] - lr_result.slope)))