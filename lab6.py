# 6.034 Lab 6 2015: Neural Nets & SVMs #todo

from nn_problems import *
from math import e

#### NEURAL NETS ###############################################################

# Wiring a neural net

nn_half = []

nn_angle = []

nn_cross = []

nn_stripe = []

nn_hexagon = []

# Optional problem; change TEST_NN_GRID to True to test locally
TEST_NN_GRID = False
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

# Helper functions
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    raise NotImplementedError

def node_value(node, input_values, neuron_outputs):
    """Given a node in the neural net, as well as a dictionary 
    of neural net input values and a dictionary mapping neuron 
    names to their outputs, computes the effective value of this 
    node."""
    if isinstance(node, basestring):
        return input_values[node] if node in input_values else neuron_outputs[node]
    else:
        return node

# Forward propagation
def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
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

def back_prop(net, input_values, desired_output, r=1, accuracy_threshold=-.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses sigmoid
    function to compute output.  Returns a tuple containing:
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
