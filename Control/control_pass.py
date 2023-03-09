import numpy as np
import typing

# Show current attributes of network
def show():
    print("=====================================")
    print("\nW weights")
    for i in range(0,len(weights)):
        print("Layer" ,i, ": " , str(weights[i]))
    
    print("\nξ neuron_potentials")
    for i in range(0,len(neuron_potentials)):
        print("Layer",i, ": " , str(neuron_potentials[i]))

    print("\nh neuron_values")
    for i in range(0,len(neuron_values)):
        print("Layer" ,i, ": " , str(neuron_values[i]))

    print("\n⌘h ew_helper_neurons")
    for i in range(0,len(ew_helper_neurons)):
        print("Layer" ,i, ": " , str(ew_helper_neurons[i]))
    
    print("\n∇h neuron_grads")
    for i in range(0,len(neuron_grads)):
        print("Layer" ,i, ": " , str(neuron_grads[i]))
    
    print("\n⌘w ew_helper_weights")
    for i in range(0,len(ew_helper_weights)):
        print("Layer" ,i, ": " , str(ew_helper_weights[i]))
    
    print("\n∇w weights_grads")
    for i in range(0,len(weights_grads)):
        print("Layer" ,i, ": " , str(weights_grads[i]))
    print("=====================================")


# Activation functions
def relu(matrix):
    return np.where(matrix <= 0, 0, matrix)

def deriv_relu(matrix):
    return np.where(matrix <= 0, 0, 1)


# Network attributs (2,3,3)
weights = np.array([
    np.ones((2,4)),
    np.ones((4,2))
    ])

ew_helper_neurons = [[],[],[]]
ew_helper_weights = [[],[],[]]
neuron_grads = [[],[],[]]
weights_grads = [[],[],[]]

neuron_potentials = []
neuron_values = []
error = np.array([])


# Data
input_vector = np.array([1,1])
target_vector = np.array([1,0])



# ---------- FORWARD PASS---------- #

# input -> hiddden
neuron_potentials.append(np.matmul(input_vector, weights[0]))
neuron_values.append(relu(neuron_potentials[0]))

# hidden -> output
neuron_potentials.append(np.matmul(neuron_values[0], weights[1]))
neuron_values.append(relu(neuron_potentials[1]))



# ---------- BACKPROPAGATION ---------- #

# Error
error = np.subtract(neuron_values[1], target_vector)

# Output layer
neuron_grads[-1] = error

# Hidden layer
ew_helper_neurons[-2] = np.multiply(
                        neuron_grads[-1],
                        deriv_relu(neuron_potentials[-1])
                    )

neuron_grads[-2] = np.matmul(
                        ew_helper_neurons[-2],
                        weights[-1].transpose()
                    )

# Input layer
ew_helper_neurons[-3] = np.multiply(
                        neuron_grads[-2],
                        deriv_relu(neuron_potentials[-2])
                    )

neuron_grads[-3] = np.matmul(
                        ew_helper_neurons[-3],
                        weights[-2].transpose()
                    )


# ---------- WEIGHTS GRADIENTS ---------- #

# Hiddnen <==> Output
ew_helper_weights[-1] = np.multiply(
                            neuron_grads[-1],
                            deriv_relu(neuron_potentials[-1])
                    )

weights_grads[-1] = np.multiply.outer(
                            ew_helper_weights[-1],
                            neuron_values[-2]
                    )

# Hidden <==> Input
ew_helper_weights[-2] = np.multiply(
                            neuron_grads[-2],
                            deriv_relu(neuron_potentials[-2])
                    )

weights_grads[-2] = np.multiply.outer(
                            ew_helper_weights[-2],
                            input_vector
                    )


# ---------- CHANGE WEIGHTS ---------- #

weights[0] = np.subtract(weights[0], weights_grads[-1])
weights[1] = np.subtract(weights[1], weights_grads[-2])


show()

print("Error before changing the weights: " + str(error))