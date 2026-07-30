import math
import numpy as np
import sympy as sp

# -----------------------------
# Basic Python math
# -----------------------------
print("math.sqrt(4):", math.sqrt(4))

# NumPy array
x_array = np.array([1, 2, 3])

# -----------------------------
# SymPy basics
# -----------------------------
print("\nsqrt(34):", sp.sqrt(34))
print("sqrt(18) ≈", sp.N(sp.sqrt(18), 8))

# Create symbolic variables
x, y = sp.symbols('x y')

# Expression
expr = 2*x**2 - x*y
print("\nExpression:")
print(expr)

# Manipulate expression
expr_manip = x * (expr + x*y + x**3)
print("\nExpression Manipulated:")
print(expr_manip)

print("\nExpanded:")
print(sp.expand(expr_manip))

print("\nFactored:")
print(sp.factor(sp.expand(expr_manip)))

# Substitute values
print("\nEvaluate expression at x=-1, y=2:")
print(expr.evalf(subs={x: -1, y: 2}))

# -----------------------------
# Symbolic function
# -----------------------------
f_symb = x**2

print("\nEvaluate x^2 at x=3:")
print(f_symb.evalf(subs={x: 3}))

# Convert symbolic function to NumPy function
f_numpy = sp.lambdify(x, f_symb, "numpy")

print("\nEvaluate x^2 on NumPy array:")
print("x =", x_array)
print("f(x) =", f_numpy(x_array))

# -----------------------------
# Symbolic differentiation
# -----------------------------
print("\nDerivative of x^3:")
print(sp.diff(x**3, x))

dfdx_composed = sp.diff(sp.exp(-2*x) + 3*sp.sin(3*x), x)

print("\nDerivative of exp(-2x)+3sin(3x):")
print(dfdx_composed)

dfdx_symb = sp.diff(f_symb, x)
print("\nDerivative of x^2:")
print(dfdx_symb)

# Convert derivative to NumPy function
dfdx_numpy = sp.lambdify(x, dfdx_symb, "numpy")

print("\nEvaluate derivative on NumPy array:")
print("x =", x_array)
print("f'(x) =", dfdx_numpy(x_array))

# -----------------------------
# Numerical Differentiation
# -----------------------------
def f(x):
    return x**2

x_array_2 = np.linspace(-5, 5, 100)

dfdx_numerical = np.gradient(f(x_array_2), x_array_2)

print("\nNumerical derivative of x^2:")
print(dfdx_numerical)

# -----------------------------
# Another function
# -----------------------------
def f_composed(x):
    return np.exp(-2*x) + 3*np.sin(3*x)

dfdx_composed_num = np.gradient(f_composed(x_array_2), x_array_2)

print("\nNumerical derivative of exp(-2x)+3sin(3x):")
print(dfdx_composed_num)

# -----------------------------
# Absolute value derivative
# -----------------------------
def dfdx_abs(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return None

print("\nDerivative of |x|:")
for value in [-3, 0, 5]:
    print(f"x = {value}, derivative = {dfdx_abs(value)}")