import math
import numpy as np
print(math.sqrt(4))


x_array = np.array([1,2,3]) 

from sympy import *

print(sqrt(34))
print(N(sqrt(18), 8))

x, y = symbols('x y')

expr = 2*x**2 - x*y
print(expr)

expr_manip = x*(expr + x*y + x**3)
print(expr_manip)

print(expand(expr_manip))
print(factor(expr_manip))

print(expr.evalf(subs={x: -1, y: 2}))

f_symb = x**2
print(f_symb.evalf(subs={x: 3}))

try:
    f_symb(x_array)
except TypeError as err:
    print(err)

diff(x**3,x)

dfdx_composed = diff(exp(-2*x) + 3*sin(3*x), x)
print(dfdx_composed)

dfdx_symb = diff(f_symb, x)
dfdx_symb_numpy = lambdify(x, dfdx_symb, 'numpy')

print("x: \n", x_array)
print("f'(x) = 2x: \n", dfdx_symb_numpy(x_array))

#Only disadvantage is sympy is not frendly to output expressions and not possible to evaluate

#Numerical Differentiation

x_array_2 = np.linspace(-5, 5, 100)
dfdx_numerical = np.gradient(f(x_array_2),x_array_2)



def f_composed(x):
    return np.exp(-2*x) + 3*np.sin(3*x)

def dfdx_abs(x):
    if x > 0:
        return 1
    else:
        if x < 0:
            return -1
        else:
            return None