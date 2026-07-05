import numpy as np

x = [1,-2,-5]
y = [4,3,-1]

def dot(x,y):
    s = 0
    for xi, yi in zip(x,y):
        s += xi * yi
    return s

#Normal Dot product function
print("The dot product of x and y is : ", dot(x,y))

#Numpy Dot product
print("The Dot product of x and y", np.dot(x,y))

#Numpy using @
print("The dot prodcut of x and y :" , np.array(x)@np.array(y))

try:
    print(x@y)
except TypeError as err: 
    print(err)

#Speed of Calculations - You need more optimized versions to solve the High Dimensional Vectors

a = np.random.rand(100000)
b = np.random.rand(100000)

import time 

tic = time.time()
c = dot(a,b)
toc = time.time()
print("Dot product: " c)
print(" Time for loop version " + str(1000*(toc-tic))+"ms" )

#In Vectorized form
tic = time.time()
c = np.dot(a,b)
toc = time.time()
print("Dot product: ", c)
print("Time for vectorized "+ str(1000*(toc - tic)+"ms"))


#In @ operator

tic = time.time()
c = a@b
toc = time.time()
print("Dot product: ",c)
print("Time for vectorized version : ", str(1000*(toc-tic))+"ms")
