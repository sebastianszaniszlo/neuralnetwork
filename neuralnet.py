import numpy as np

def nonlin(x, deriv=False):
    if(deriv==True):
        return x * (1 - x);

    return 1 / (1 + np.exp(-x));

# input data
x = np.array([[0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1]])

# output data
y = np.array([[0],
            [1],
            [1],
            [0]])

np.random.seed(1)

# synapses

syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

# training step
for i in range(60000):
    l0 = x
    # multiply the input with the synapse matrix and apply the activation function on it
    l1 = nonlin(np.dot(l0, syn0))
    # multiply the predicted data( from the hidden layer ) with the second synapse matrix and run it through the activation function
    l2 = nonlin(np.dot(l1, syn1));
    
    # calculate the error by subtracting the predicted output from the actual output
    l2_error = y - l2

    # print the error at a given interval
    if (i % 1000) == 0:
        print("Error: " + str(np.mean(np.abs(l2_error))))

    # back-propagation

    # multiply the l2 error with the sigmoid derivative
    l2_delta = l2_error * nonlin(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1, deriv=True)

    # update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print("Predicted output: ")
print(l2)
print("Actual output: ")
print(y)