import numpy as np

def f(x):
    return x**2
print(f(3))


def dfdx(x):
    return 2*x
print(dfdx(3))

x_array = np.array([1,2,3])
print("x: ",x_array)
print("f(x) : ",f(x_array))
print("dfdx f(x): ",dfdx(x_array))