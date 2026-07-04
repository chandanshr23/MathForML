import numpy as np

A = np.array([[4,-3,1],[2,1,3],[-1,2,-5]],dtype = np.dtype(float))
b = np.array([-10,0,17],dtype = np.dtype(float))

print(f"MAtrix of A : {A} and shape : {np.shape(A)}")
print(f"Matrix of B : {b} and shape : {np.shape(b)}")

A_res = np.linalg.solve(A,b)
A_det = np.linalg.det(A)

print(f"Sovled Equations : {A_res}")
print(f"Determinant of A : {A_det}")