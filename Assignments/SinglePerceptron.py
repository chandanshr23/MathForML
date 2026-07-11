import numpy as np
import matplotlib.pyplot as  plt
import pandas as pd
import tools

from sklearn.datasets import make_regression

np.random.seed(3)

m = 30
X, Y = make_regression(n_samples=m , n_features=1, noise=20, random_state=1)
X = X.reshape((1,m))
Y = Y.reshape((1,m))

print('Training dataset X = ')
print(X)
print('Training on Y dataset = ')
print(Y)

plt.scatter(X, Y , c="black")
plt.xlabel("$x$")
plt.ylabel("$y$")
#plt.show()


#To find the shape of the data training examples
X_shape = X.shape
Y_shape = Y.shape

m = X.size

print(' The shape of X'+ str(X_shape))
print(' The shape of y '+str(Y_shape))


#Implementation of Neural Network

def layer_sizes(X , Y):
    """
    Arguments :
    X -- input dataset of shape (input size, numbero of examples)
    Y -- output dataset of shape (output size ,number of examples)

    Returns:
    n_x= the size of the input layer
    n_y = The size of the output layer

    """

    n_x = X.shape[0]
    n_y = Y.shape[0]

    return (n_x, n_y)

(n_x, n_y) = layer_sizes(X, Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the output layer is: n_y = " + str(n_y))

def initialize_parameters(n_x , n_y):
    """
    Returns :
    params -- pyton dictionary containing your parameters:
                W -- weights matrix of shape (n_y,n_X)
                b -- bias value set as vector of shape (n_y,1)
    """
    W = np.random.randn(n_y, n_x)*0.01
    b = np.zeros((n_y,1))

    assert (W.shape ==(n_y,n_x))
    assert (b.shape ==(n_y,1))

    parameters = {"W":W, "b":b}

    return parameters

parameters = initialize_parameters(n_x, n_y)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))

#Implement Forward Propagation

def forward_propogation(X, parameters):
    """
    Arguments :
    X -- input data of size (n_x ,m)
    parameters -- python dictionary containing your parameters 

    Returns:
        Y_hat = the outpu
    """
    W = parameters["W"]
    b = parameters["b"]

    Z = np.dot(W,X)+b
    Y_hat = Z

    assert(Y_hat.shape == (n_y,X.shape[1]))
    return Y_hat

Y_hat = forward_propogation(X,parameters)
print(Y_hat)

#Defining Cost function to train model 

def compute_cost(Y_hat, Y):
    """
    Computes the cost function as the sum of squares
    Arguments :
    Y_hat -- the output of the neural networks of the shape 
    Y -- "true" labels vector of shape 

    Returns : 

    cost -- sum of the squares scaled by 1/(2* number of examples)
    """
    m = Y.shape[1]

    cost = np.sum((Y_hat - Y)**2)/(2*m)
    return cost 

print ("the cost : "+str(compute_cost(Y_hat, Y)))

#To minimize the cost function like bringining as close to 0 , To Achieve this
#backward prop needs to be performed

parameters = tools.train_nn(parameters, Y_hat,X,Y)
print("W=" + str(parameters["W"]))
print("b= "+str(parameters["b"]))


#Integrate all the Forward Propogation,Cost function, Neural Neetwork in the NN model

def nn_model(X, Y, num_iteration=10, print_cost=True):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]
    parameters = initialize_parameters(n_x, n_y)

    for i in range(0, num_iteration):
        Y_hat = np.dot(parameters["W"], X) + parameters["b"]
        cost = np.sum((Y_hat - Y)**2)/(2*Y.shape[1])
        parameters = tools.train_nn(parameters, Y_hat, X, Y)

        if i == 0:  # only first iteration, don't flood output
            print("Y_hat sample:", Y_hat[:, :5])
            print("cost:", cost)
            print("W after update:", parameters["W"])
            print("b after update:", parameters["b"])

        if print_cost:
            print("Cost after the iteration %i: %f" % (i, cost))

    return parameters

parameters = nn_model(X,Y,num_iteration=15,print_cost=True)
print("W= "+ str(parameters["W"]))
print("b= "+str(parameters["b"]))

W_simple = parameters["W"]
b_simple = parameters["b"]

X_pred = np.array([-0.95, 0.2, 1.5])

fig, ax = plt.subplots()
plt.scatter(X, Y, color ="black")
plt.xlabel("$x$")
plt.ylabel("$y$")

X_line = np.arange(np.min(X[0:]),np.max(X[0,:])*1.1,0.1)
ax.plot(X_line,W_simple[0,0]*X_line+b_simple[0,0],"r")
ax.plot(X_pred, W_simple[0,0]*X_pred+b_simple[0,0],"bo")
# plt.show()
# plt.show()

#Muliple Regression Model

df = pd.read_csv(r'C:\Users\2026\Desktop\MathForML\Assignments\data\house_prices_train.csv')

X_multi = df[['GrLivArea','OverallQual']]
Y_multi = df['SalePrice']

print(f"X_multi: \n{X_multi}\n")
print(f"Y_multi : \n{Y_multi}\n")
print(df[['GrLivArea','OverallQual','SalePrice']].isna().sum())
print(df.shape)
print(X_multi.dtypes)

#Normalization : subtract mean of the array from each of the elements and divide them by Standard deviation
X_multi_norm = (X_multi - X_multi.mean()) / X_multi.std()
Y_multi_norm = (Y_multi - Y_multi.mean())/Y_multi.std()

X_multi_norm = np.array(X_multi_norm).T
Y_multi_norm = np.array(Y_multi_norm).reshape((1,len(Y_multi_norm)))

print('The shape of x : '+str(X_multi_norm.shape))
print('The shape of Y: '+str(Y_multi_norm.shape))
print('The m = %d training examples' %(X_multi_norm.shape[1]))


#Testing with the Neural model above 
parameters_multi = nn_model(X_multi_norm, Y_multi_norm,num_iteration=100,print_cost=True)

print("W = "+str(parameters_multi["W"]))
print("b= "+str(parameters_multi["b"]))

W_multi = parameters_multi["W"]
b_multi = parameters_multi["b"]
# 1. Raw correlation before any normalization — ground truth to compare against
print(X_multi.corrwith(Y_multi))

# 2. Confirm both normalized columns actually look like mean 0, std 1
print("GrLivArea norm - mean:", X_multi_norm[0].mean(), "std:", X_multi_norm[0].std())
print("OverallQual norm - mean:", X_multi_norm[1].mean(), "std:", X_multi_norm[1].std())

# 3. Peek at actual normalized values for OverallQual — is it degenerate/constant?
print(X_multi_norm[1, :10])

X_pred_multi = np.array([[1710, 7], [1200, 6], [2200, 8]]).T

# Normalize using the same mean and standard deviation of the original training array X_multi.
X_multi_mean = np.array(X_multi.mean()).reshape((2,1))
X_multi_std = np.array(X_multi.std()).reshape((2,1))
X_pred_multi_norm = (X_pred_multi - X_multi_mean)/ X_multi_std
# Make predictions.
Y_pred_multi_norm = np.matmul(W_multi, X_pred_multi_norm) + b_multi
# Denormalize using the same mean and standard deviation of the original training array Y_multi.
Y_pred_multi = Y_pred_multi_norm * np.std(Y_multi) + np.mean(Y_multi)

print(f"Ground living area, square feet:\n{X_pred_multi[0]}")
print(f"Rates of the overall quality of material and finish, 1-10:\n{X_pred_multi[1]}")
print(f"Predictions of sales price, $:\n{np.round(Y_pred_multi)}")