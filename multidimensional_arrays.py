import numpy as np

# Create a 2 dimensional array (2-D)
two_dim_arr = np.array([[1,2,3], [4,5,6]])
print(two_dim_arr)

#Another way 
one_dimensional = np.array([1,2,3,4,5,6])
multi_dimensional = np.reshape(one_dimensional,(2,3))
print(multi_dimensional)

#Shapes,Size,dimension
multi_dimensional.ndim
multi_dimensional.shape
multi_dimensional.size


#Maths
arr1=np.array([1,2,3])
arr2=np.array([4,5,6])
addition=arr1+arr2
subtration=arr1-arr2
multiplication=arr1*arr2

#Vector and Multiplication

vector=arr1*1.6



#Indexing
# Select the third element of the array. Remember the counting starts from 0.
a = ([1, 2, 3, 4, 5])
print(a[2])

# Select the first element of the array.
print(a[0])

# Indexing on a 2-D array
two_dim = np.array(([1, 2, 3],
          [4, 5, 6], 
          [7, 8, 9]))

# Select element number 8 from the 2-D array using indices i, j.
print(two_dim[2][1])


#Slicing
sliced_arr = a[1:4]
print(sliced_arr)
# Slice the array a to get the array [1,2,3]
sliced_arr = a[:3]
print(sliced_arr)
# Slice the array a to get the array [1,3,5]
sliced_arr = a[::2]
print(sliced_arr)
# Note that a == a[:] == a[::]
print(a == a[:] == a[::])
# Slice the two_dim array to get the first two rows
sliced_arr_1 = two_dim[0:2]
sliced_arr_1
# Similarily, slice the two_dim array to get the last two rows
sliced_two_dim_rows = two_dim[1:3]
print(sliced_two_dim_rows)
sliced_two_dim_cols = two_dim[:,1]
print(sliced_two_dim_cols)


#Stacking

a1 = np.array([[1,1], 
               [2,2]])
a2 = np.array([[3,3],
              [4,4]])
print(f'a1:\n{a1}')
print(f'a2:\n{a2}')

# Stack the arrays vertically
vert_stack = np.vstack((a1, a2))
print(vert_stack)

# Stack the arrays horizontally
horz_stack = np.hstack((a1, a2))
print(horz_stack)