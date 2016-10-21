# MIT 6.034 Lab 6: Neural Nets
# Written by Jessica Noss (jmn), Dylan Holmes (dxh), Jake Barnwell (jb16), and 6.034 staff

from nn_problems import *
from math import e

#### NEURAL NETS ###############################################################

# Wiring a neural net

nn_half = []

nn_angle = []

nn_cross = []

nn_stripe = []

nn_hexagon = []

nn_grid = []

# Thresholding functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    raise NotImplementedError

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    raise NotImplementedError

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    raise NotImplementedError

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    raise NotImplementedError

# Forward propagation
def node_value(node, input_values, neuron_outputs):
    # Optional helper function; might be helpful later on
    """Given a node in the neural net, as well as a dictionary
    of neural net input values and a dictionary mapping neuron
    names to their outputs, computes the effective value of this
    node."""
    raise NotImplementedError

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    raise NotImplementedError

# Backward propagation warmup
def gradient_step(func, values, delta):
    """Given some unknown function of three variables and a list of three
    values representing the current inputs into the function,
    finds the amount that the function changes
    by varying (or not) each of the input variables by +/- delta (a total
    3^3 = 27 possible assignments for the three variables). Picks
    the assignments of variables that yields the smallest result when
    input into the function, and returns a tuple containing
    (1) the function value at the lowest point found, and
    (2) the list of variable assignments that yielded the lowest
    function value."""
    raise NotImplementedError

def calculate_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons,
    and Wires whose outputs/values are required to compute the
    delta_B coefficient required to update this wire's weight."""
    raise NotImplementedError

# Backward propagation
def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    raise NotImplementedError

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    raise NotImplementedError

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    raise NotImplementedError


#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
