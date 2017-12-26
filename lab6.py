# MIT 6.034 Lab 6: Neural Nets

from neural_net_api import *
from math import e
INF = float('inf')


# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    if x >= threshold:
        return 1
    else:
        return 0

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1/(1+e**(-1*steepness*(x-midpoint)))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return max(0, x)

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -0.5*(desired_output-actual_output)**2


#### Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  
    """
    Given 
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))
    
    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node
    
    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    neurons = net.topological_sort()
    mapped = input_values.copy()
    finalOutput = 0
    for neuron in neurons:
        allIncoming = net.get_incoming_neighbors(neuron)
        sum = 0
        for incoming in allIncoming:
            for wire in net.get_wires(incoming, neuron):
                outputVal = node_value(incoming, mapped, mapped)
                sum += (outputVal*wire.get_weight())

        mapped[neuron] = threshold_fn(sum)
        if net.is_output_neuron(neuron):
            finalOutput = mapped[neuron]
    return (finalOutput, mapped)


#### Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    perturbations = [step_size, -1*step_size, 0]
    maxi = -1*INF
    highest = []
    for pertub1 in perturbations:
        for pertub2 in perturbations:
            for pertub3 in perturbations:
                val = func(inputs[0] + pertub1, inputs[1] + pertub2, inputs[2] + pertub3)
                if val > maxi:
                    maxi = val
                    highest = [inputs[0] + pertub1, inputs[1] + pertub2, inputs[2] + pertub3]
    return (maxi, highest)

def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    toReturn = set()
    toReturn.add(wire.startNode)
    neurons = [wire.endNode]
    for wire in net.get_wires(wire.startNode, wire.endNode):
        toReturn.add(wire)
    while len(neurons) > 0:
        neuron = neurons.pop(0)
        toReturn.add(neuron)
        if net.is_output_neuron(neuron):
            return toReturn

        for outgoing in net.get_outgoing_neighbors(neuron):
            neurons.append(outgoing)
            for wire in net.get_wires(neuron, outgoing):
                toReturn.add(wire)
    return toReturn


def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    updateCoeff = {}

    for neuron in net.topological_sort()[::-1]:
        out_B = neuron_outputs[neuron]
        if net.is_output_neuron(neuron):
            delta_B = out_B * (1 - out_B) * (desired_output - out_B)
            updateCoeff[neuron] = delta_B
        else:
            sumAll = 0
            for outgoing in net.get_outgoing_neighbors(neuron):
                for wire in net.get_wires(neuron, outgoing):
                    sumAll += wire.weight * updateCoeff[wire.endNode]
            delta_B = out_B * (1 - out_B) * sumAll
            updateCoeff[neuron] = delta_B

    return updateCoeff

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    updateCoeff = calculate_deltas(net, desired_output, neuron_outputs)

    for wire in net.get_wires():
        wire.set_weight(wire.get_weight() + r * node_value(wire.startNode, input_values, neuron_outputs) * updateCoeff[wire.endNode])

    return net


def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    count = 0
    neuron_outputs = forward_prop(net, input_values, sigmoid)[1]
    actual_output = forward_prop(net, input_values, sigmoid)[0]

    newNet = net
    while(accuracy(desired_output, actual_output) < minimum_accuracy):
        newNet = update_weights(net, input_values, desired_output, neuron_outputs, r)
        count += 1
        neuron_outputs = forward_prop(net, input_values, sigmoid)[1]
        actual_output = forward_prop(net, input_values, sigmoid)[0]

    return (newNet, count)


