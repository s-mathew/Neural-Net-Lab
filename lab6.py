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

# Optional problem; change TEST_NN_GRID to True to test locally
TEST_NN_GRID = False
nn_grid = []

# Helper functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    raise NotImplementedError

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    raise NotImplementedError

def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    raise NotImplementedError

# Forward propagation
def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    raise NotImplementedError

# Backward propagation
def calculate_deltas(net, input_values, desired_output):
    """Computes the update coefficient (delta_B) for each neuron in the
    neural net.  Uses sigmoid function to compute output.  Returns a dictionary
    mapping neuron names to update coefficient (delta_B values)."""
    raise NotImplementedError

def update_weights(net, input_values, desired_output, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses
    sigmoid function to compute output.  Returns the modified neural net, with
    updated weights."""
    raise NotImplementedError

def back_prop(net, input_values, desired_output, r=1, accuracy_threshold=-0.001):
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
