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

x=np.linalg.solve(A,b)
print(f"Soltion of A and B :{x}")

det = np.linalg.det(A)
print(f"Determinant of Matrix A: {det:.2f}")