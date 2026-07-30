import numpy as np

A = np.array([[4,-3,1],[2,1,3],[-1,2,-5]],dtype = np.dtype(float))
b = np.array([-10,0,17],dtype = np.dtype(float))

print(f"MAtrix of A : {A} and shape : {np.shape(A)}")
print(f"Matrix of B : {b} and shape : {np.shape(b)}")

A_res = np.linalg.solve(A,b)
A_det = np.linalg.det(A)

print(f"Sovled Equations : {A_res}")
print(f"Determinant of A : {A_det}")



#To Solve the matrix using Row Reduction Technique
A_system = np.hstack((A, b.reshape((3, 1))))

print(A_system)

# exchange row_num of the matrix M with its multiple by row_num_multiple
# Note: for simplicity, you can drop check if  row_num_multiple has non-zero value, which makes the operation valid
def MultiplyRow(M, row_num, row_num_multiple):
    # .copy() function is required here to keep the original matrix without any changes
    M_new = M.copy()
    M_new[row_num] = M_new[row_num] * row_num_multiple
    return M_new

print("Original matrix:")
print(A_system)
print("\nMatrix after its third row is multiplied by 2:")
# remember that indexing in Python starts from 0, thus index 2 will correspond to the third row
print(MultiplyRow(A_system,2,2))

# multiply row_num_1 by row_num_1_multiple and add it to the row_num_2, 
# exchanging row_num_2 of the matrix M in the result
def AddRows(M, row_num_1, row_num_2, row_num_1_multiple):
    M_new = M.copy()
    M_new[row_num_2] = row_num_1_multiple * M_new[row_num_1] + M_new[row_num_2]
    return M_new

print("Original matrix:")
print(A_system)
print("\nMatrix after exchange of the third row with the sum of itself and second row multiplied by 1/2:")
print(AddRows(A_system,1,2,1/2))

# exchange row_num_1 and row_num_2 of the matrix M
def SwapRows(M, row_num_1, row_num_2):
    M_new = M.copy()
    M_new[[row_num_1, row_num_2]] = M_new[[row_num_2, row_num_1]]
    return M_new

print("Original matrix:")
print(A_system)
print("\nMatrix after exchange its first and third rows:")
print(SwapRows(A_system,0,2))

A_ref = SwapRows(A_system,0,2)
# Note: ref is an abbreviation of the row echelon form (row reduced form)
print(A_ref)

# multiply row 0 of the new matrix A_ref by 2 and add it to the row 1
A_ref = AddRows(A_ref,0,1,2)
print(A_ref)

A_ref = AddRows(A_ref,0,2,4)
print(A_ref)

# multiply row 1 of the new matrix A_ref by -1 and add it to the row 2
A_ref = AddRows(A_ref,1,2,-1)
print(A_ref)

# multiply row 2 of the new matrix A_ref by -1/12
A_ref = MultiplyRow(A_ref,2,-1/12)
print(A_ref)

# multiply row 2 of the new matrix A_ref by -1/12
A_ref = MultiplyRow(A_ref,2,-1/12)
print(A_ref)