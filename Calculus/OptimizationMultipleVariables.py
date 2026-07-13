import numpy as np
import matplotlib.pyplot as plt
from tools.w2_tools import (plot_f_cont_and_surf, gradient_descent_two_variables, 
                            f_example_3,dfdx_example_3,dfdy_example_3,f_example_4,dfdx_example_4 , dfdy_example_4)

plot_f_cont_and_surf([0,5],[0,5],[74,85],f_example_3,cmap='coolwarm',view={'azim':-60,'elev':28})

def gradient_descent(dfdx,dfdy,x,y,learning_Rate=0.1,num_iterations=100):
    for iteration in range(num_iterations):
        x, y = x - learning_Rate*dfdx(x,y), y -learning_Rate*dfdy(x,y)
    return x, y

num_iterations = 30; learning_rate = 0.25; x_initial = 0.5; y_initial = 0.6

print("Gradient descent result : x_min and y_min: ",gradient_descent(dfdx_example_3,dfdy_example_3, x_initial,y_initial, learning_rate, num_iterations))

num_iterations = 20; learning_rate =0.25; x_initial = 0.5; y_initial = 0.6

gd_example_3 = gradient_descent_two_variables([0,5],[0,5],[74,85], f_example_3,dfdx_example_3,dfdy_example_3,
                                              gradient_descent, num_iterations, learning_rate,
                                              x_initial,y_initial,[0.1,0.1,81.5],2,[4,1,171],cmap='coolwarm', view={'azim':-60,'elev':28})

plot_f_cont_and_surf([0, 5], [0, 5], [6, 9.5], f_example_4, cmap='terrain', view={'azim':-63,'elev':21})


#To find minima of the given parameters using gradient descent
num_iterations = 100; learning_rate = 0.2; x_initial = 0.5; y_initial = 3

print("Gradient descent result: x_min, y_min =", 
      gradient_descent(dfdx_example_4, dfdy_example_4, x_initial, y_initial, learning_rate, num_iterations)) 

# Converges to the global minimum point.
num_iterations = 30; learning_rate = 0.2; x_initial = 0.5; y_initial = 3
# Converges to a local minimum point.
# num_iterations = 20; learning_rate = 0.2; x_initial = 2; y_initial = 3
# Converges to another local minimum point.
# num_iterations = 20; learning_rate = 0.2; x_initial = 4; y_initial = 0.5

gd_example_4 = gradient_descent_two_variables([0, 5], [0, 5], [6, 9.5], 
                                              f_example_4, dfdx_example_4, dfdy_example_4, 
                                              gradient_descent, num_iterations, learning_rate, 
                                              x_initial, y_initial, 
                                              [2, 2, 6], 0.5, [2, 1, 63], 
                                              cmap='terrain', view={'azim':-63,'elev':21})