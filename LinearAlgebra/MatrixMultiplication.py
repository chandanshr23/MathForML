import numpy as np

#Matrix multiplication 

A = np.array([[4,9,9],[9,1,6],[9,2,3]])
B = np.array([[2,2],[5,7],[4,4]])

#Result 
C = np.matmul(A,B)

print(C)

#even A@B works the same way 

C_res = A @ B

#Matrix Convention and Broadcasting
try:
    np.matmul(B, A)
except ValueError as err:
    print(err)
try:
    B @ A
except ValueError as err:
    print(err)

x = np.array([1, -2, -5])
y = np.array([4, 3, -1])

print("Shape of vector x:", x.shape)
print("Number of dimensions of vector x:", x.ndim)
print("Shape of vector x, reshaped to a matrix:", x.reshape((3, 1)).shape)
print("Number of dimensions of vector x, reshaped to a matrix:", x.reshape((3, 1)).ndim)

np.matmul(x,y)

try:
    np.matmul(x.reshape((3, 1)), y.reshape((3, 1)))
except ValueError as err:
    print(err)

np.dot(A, B)


#Implementing Matrix Multiplication from scratch Matmul 

import numpy as np

def my_matmul(A, B):

    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    if cols_A != rows_B:
        raise ValueError("Matrix dimensions don't match.")

    C = np.zeros((rows_A, cols_B))

    for i in range(rows_A):

        for j in range(cols_B):

            total = 0

            for k in range(cols_A):

                total += A[i][k] * B[k][j]

            C[i][j] = total

    return C

#Verify
A = np.array([[4,9,9],
              [9,1,6],
              [9,2,3]])

B = np.array([[2,2],
              [5,7],
              [4,4]])

print(my_matmul(A,B))
print(np.matmul(A,B))   