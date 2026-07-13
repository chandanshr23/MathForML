import numpy as np
import matplotlib.pyplot as plt 
from tools.w2_tools import gradient_descent_one_variable, plot_f, f_example_2,dfdx_example_2

def f_example_1(x):
    return np.exp(x)-np.log(x)

def dfdx_example(x):
    return np.exp(x)-1/x

def gradient_descent(dfdx,x,learning_rate=0.1,num_iterations=100):
    for iteration in range(num_iterations):
        x = x - learning_rate*dfdx(x)
    return x

num_iterations = 25; learning_rate =0.1 ; x_initial =1.6

print("Gradient descent result : x_min =", gradient_descent(dfdx_example, x_initial, learning_rate, num_iterations))

num_iterations = 25; learning_rate = 0.1; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.3; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.5; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.04; x_initial = 1.6
# num_iterations = 75; learning_rate = 0.04; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.1; x_initial = 0.05
# num_iterations = 25; learning_rate = 0.1; x_initial = 0.03
# num_iterations = 25; learning_rate = 0.1; x_initial = 0.02

gd_example_1 = gradient_descent_one_variable([0.001, 2.5], [-0.3, 13], f_example_1, dfdx_example, 
                                   gradient_descent, num_iterations, learning_rate, x_initial, 0.0, [0.35, 9.5])

plot_f([0.001, 2], [-6.3, 5], f_example_2, -6)

print("Gradient descent results")
print("Global minimum: x_min =", gradient_descent(dfdx_example_2, x=1.3, learning_rate=0.005, num_iterations=35)) 
print("Local minimum: x_min =", gradient_descent(dfdx_example_2, x=0.25, learning_rate=0.005, num_iterations=35)) 

num_iterations = 35; learning_rate = 0.005; x_initial = 1.3
# num_iterations = 35; learning_rate = 0.005; x_initial = 0.25
# num_iterations = 35; learning_rate = 0.01; x_initial = 1.3

gd_example_2 = gradient_descent_one_variable([0.001, 2], [-6.3, 5], f_example_2, dfdx_example_2, 
                                      gradient_descent, num_iterations, learning_rate, x_initial, -6, [0.1, -0.5])