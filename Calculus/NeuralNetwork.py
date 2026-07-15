#Solving simple classification problem

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from sklearn.datasets import make_blobs 

np.random.seed(3)

fig, ax= plt.subplots()
xmin, xmax = -0.2, 1.4
x_line = np.arange(xmin, xmax, 0.1)

ax.scatter(0, 0 , color="b")
ax.scatter(0, 1, color='r')
ax.scatter(1, 0, color='b')
ax.scatter(1, 1, color='b')
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

ax.plot(x_line, x_line+0.5, color="black")
plt.plot()


#Implement a Single Neural Network with activation function

m = 30
X = np.random.randint(0, 2, (2, m))
Y = np.logical_and(X[0] == 0,  X[1] == 1).astype(int).reshape((1, m))

print('Training dataset X x1 and x2 coordinates ')
print(X)
print('Trainiong dataset Y bluew and red')
print(Y)
print('The shape of X : ' + str(X.shape))
print('The shape of Y :' + str(Y.shape))
print(' I have m =%d training examples' %(X.shape[1]))

#The Activateion Function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

print("sigmoid(-2) = " + str(sigmoid(-2)))
print("sigmoid(0) = "+ str(sigmoid(0)))
print("sigomoid(3.5)= "+str(sigmoid(3.5)))
print("sigmoid(  )")
print(sigmoid(np.array([-2, 0, 3.5])))



#implement neural network
def layer_sizes(X,Y):
    """
    Arguments:
    X -- input datasets of shape (input size, number of examples)
    Y -- output the shape (output size , numbr of examples)

    REturns:
    n_x -- the size of the input layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]

    return (n_x, n_y)

(n_x, n_y) = layer_sizes(X, Y)
print("the size of the input layer is : n_x " + str(n_x))
print("the size of the output layuer is :" + str(n_y))


def initiailize_parameter(n_x, n_y):
    """
    Returns : 
    params -- python dictionaries containing parameters
                W -- weight matrix of shape(n_y, n_x)
                b -- bias value set as a vector of shape(n_y,1)
    """

    W = np.random.randn(n_y, n_x) *0.01
    b = np.zeros((n_y ,1))

    parameters = {"W": W,
                  "b": b}
    return parameters

parameters = initiailize_parameter(n_x, n_y)
print("W = "+ str(parameters["W"]))
print("b = "+ str(parameters["b"]))

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A -- The output
    """
    W = parameters["W"]
    b = parameters["b"]
    
    # Forward Propagation to calculate Z.
    Z = np.matmul(W, X) + b
    A = sigmoid(Z)

    return A

A = forward_propagation(X, parameters)

print("Output vector A:", A)

def compute_cost(A, Y):
    """
    Computes the log loss cost function
    
    Arguments:
    A -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    cost -- log loss
    
    """
    # Number of examples.
    m = Y.shape[1]

    # Compute the cost function.
    logprobs = - np.multiply(np.log(A),Y) - np.multiply(np.log(1 - A),1 - Y)
    cost = 1/m * np.sum(logprobs)
    
    return cost

print("cost = " + str(compute_cost(A, Y)))


def backward_propagation(A, X, Y):
    """
    Implements the backward propagation, calculating gradients
    
    Arguments:
    A -- the output of the neural network of shape (n_y, number of examples)
    X -- input data of shape (n_x, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # Backward propagation: calculate partial derivatives denoted as dW, db for simplicity. 
    dZ = A - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    
    grads = {"dW": dW,
             "db": db}
    
    return grads

grads = backward_propagation(A, X, Y)

print("dW = " + str(grads["dW"]))
print("db = " + str(grads["db"]))


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients 
    learning_rate -- learning rate parameter for gradient descent
    
    Returns:
    parameters -- python dictionary containing updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]
    
    # Retrieve each gradient from the dictionary "grads".
    dW = grads["dW"]
    db = grads["db"]
    
    # Update rule for each parameter.
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

parameters_updated = update_parameters(parameters, grads)

print("W updated = " + str(parameters_updated["W"]))
print("b updated = " + str(parameters_updated["b"]))

def nn_model(X, Y, num_iterations=10, learning_rate=1.2, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (n_y, number of examples)
    num_iterations -- number of iterations in the loop
    learning_rate -- learning rate parameter for gradient descent
    print_cost -- if True, print the cost every iteration
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to make predictions.
    """
    
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]
    
    parameters = initiailize_parameter(n_x, n_y)
    
    # Loop
    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A".
        A = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A, Y". Outputs: "cost".
        cost = compute_cost(A, Y)
        
        # Backpropagation. Inputs: "A, X, Y". Outputs: "grads".
        grads = backward_propagation(A, X, Y)
    
        # Gradient descent parameter update. Inputs: "parameters, grads, learning_rate". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every iteration.
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

parameters = nn_model(X, Y, num_iterations=50, learning_rate=1.2, print_cost=True)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))

def plot_decision_boundary(X, Y, parameters):
    W = parameters["W"]
    b = parameters["b"]

    fig, ax = plt.subplots()
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));
    
    x_line = np.arange(np.min(X[0,:]),np.max(X[0,:])*1.1, 0.1)
    ax.plot(x_line, - W[0,0] / W[0,1] * x_line + -b[0,0] / W[0,1] , color="black")
    plt.plot()
    plt.show()
    
plot_decision_boundary(X, Y, parameters)

def predict(X, parameters):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (blue: False / red: True)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A = forward_propagation(X, parameters)
    predictions = A > 0.5
    
    return predictions

X_pred = np.array([[1, 1, 0, 0],
                   [0, 1, 0, 1]])
Y_pred = predict(X_pred, parameters)

print(f"Coordinates (in the columns):\n{X_pred}")
print(f"Predictions:\n{Y_pred}")

#Larger DAta
# Dataset
n_samples = 1000
samples, labels = make_blobs(n_samples=n_samples, 
                             centers=([2.5, 3], [6.7, 7.9]), 
                             cluster_std=1.4,
                             random_state=0)

X_larger = np.transpose(samples)
Y_larger = labels.reshape((1,n_samples))

plt.scatter(X_larger[0, :], X_larger[1, :], c=Y_larger, cmap=colors.ListedColormap(['blue', 'red']));

parameters_larger = nn_model(X_larger, Y_larger, num_iterations=100, learning_rate=1.2, print_cost=False)
print("W = " + str(parameters_larger["W"]))
print("b = " + str(parameters_larger["b"]))

plot_decision_boundary(X_larger, Y_larger, parameters_larger)