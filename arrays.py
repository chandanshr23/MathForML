
#Numpy Library
import numpy as np

#Create an array
arr = np.array([1,1,12,3,4])
print(arr)


b = np.arange(3)
print(b)#arange means 3= [0,1,2]

c = np.arange(1,20,3)
print(c) #[ 1  4  7 10 13 16 19]


lin_spaced_arr = np.linspace(0, 100, 5)
print(lin_spaced_arr) #[  0.  25.  50.  75. 100.]

c_int = np.arange(1, 20, 3, dtype=int)
print(c_int) #[ 1  4  7 10 13 16 19]

#TOSet Float elements
b_float = np.arange(3, dtype=float)
print(b_float)

char_arr = np.array(['Welcome to Math for ML!'])
print(char_arr)
print(char_arr.dtype) # Prints the data type of the array ['Welcome to Math for ML!']nd <U23