import numpy as np

#Represent the Linear Equations
#-x_1+3x_2=7 and 3x_1+2x_2=1

A = np.array([[-1,3],
           [3,2]],dtype=np.dtype(float))
b = np.array([7,1],dtype=np.dtype(float))
print("Matrix A:")
print(A)

print("Matrix B:")
print(b)

print(f"Shape of Matrix A: {A.shape}")
print(f"Shape of matrix B: {b.shape}")

#Solve the Linear algebra using the linalg function
#in solve funciton the LU method is applied since its the fastest
x=np.linalg.solve(A,b)
print(f"Soltion of A and B :{x}")

det = np.linalg.det(A)
print(f"Determinant of Matrix A: {det:.2f}")


#Extra How solve works and Why LU decompostion is better than Gaussian Elimnation
"""
Quick reference: Gaussian Elimination vs LU Decomposition vs np.linalg.solve
-----------------------------------------------------------------------------
Core idea:
- Gaussian elimination solves Ax=b from scratch every time -> O(n^3) EACH call.
- LU factors A = L @ U once (O(n^3)), then reuses L, U for any b via
  forward/back substitution (O(n^2) each). Cheaper when solving for many b's.
- np.linalg.solve(A, b) uses LU + partial pivoting under the hood (LAPACK gesv).
  It never computes A_inverse explicitly (that's slower & less stable).
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

# -------------------------------------------------
# 1. Plain Gaussian elimination (naive, for intuition only)
# -------------------------------------------------
def gaussian_elimination(A, b):
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = len(b)

    # Forward elimination -> upper triangular
    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


# -------------------------------------------------
# 2. LU decomposition: factor once, solve many times
# -------------------------------------------------
A = np.array([[4, 3, 2],
              [2, 1, 1],
              [1, 2, 3]], dtype=float)

b1 = np.array([10, 5, 8], dtype=float)
b2 = np.array([1, 2, 3], dtype=float)   # different RHS, same A

# Factor ONCE
lu, piv = lu_factor(A)

# Reuse the factorization for every new b (cheap, O(n^2) each)
x1 = lu_solve((lu, piv), b1)
x2 = lu_solve((lu, piv), b2)

print("Gaussian result:", gaussian_elimination(A, b1))
print("LU result (b1): ", x1)
print("LU result (b2): ", x2)

# -------------------------------------------------
# 3. np.linalg.solve — the "just give me the answer" way
#    (uses LU + pivoting internally, no explicit inverse)
# -------------------------------------------------
x_np = np.linalg.solve(A, b1)
print("np.linalg.solve:", x_np)

# Avoid this (slower, numerically worse):
# x_bad = np.linalg.inv(A) @ b1