import jax.numpy as np
from activations import activation_dict
import pickle

class pillarMC:
    def __init__(self):
        self.layers = []
        self.fwdpass_output = None

    #managing addition of layers without size mismatch (morph kind of makes it easy
    def add(self,layer,morph = False):
        if len(self.layers)==0:
            self.layers.append(layer)
        else:
            prev_indim,prev_outdim = self.layers[-1].size()
            curr_indim ,curr_outdim = layer.size()
            if prev_outdim != curr_indim:
                if morph:
                    layer.resize(curr_outdim,prev_outdim)
                    self.layers.append(layer)
                else:
                    raise Exception(f'Dimensionality mismatch between consecutive layers: output dimension of previous layer {prev_outdim} != input dimension of current layer {curr_indim}. Set morph parameter to True to self adjust for previous layer')
            else:
                self.layers.append(layer)

    #handles the forward pass for all the layers
    def fwd_pass(self,input):
        a = input
        for layer in self.layers:
            a = layer.forward_pass(a)
        self.fwdpass_output = a
        return a

    #handles the backward pass for all layers
    def bkwd_pass(self , output_loss_grad):
        dell = self.layers[-1].activator_derivative(self.layers[-1].pre_activation)*output_loss_grad
        w_temp = self.layers[-1].weights
        for layer in self.layers[-2::-1]:
            dell , w_temp = layer.backward_pass(dell,w_temp)
        return dell

    #to be used for output (fwd_pass uses dropout)
    def output(self,input):
        a = input
        for layer in self.layers:
            a = layer.forward(a)
        return a

    #updates the layer parameters (weights and biases) as per the provided gradient * learning_rate
    def update_parameters(self,gradients,learning_rate=10e-3):
        w_grad,b_grad = gradients
        for i,layer in enumerate(self.layers):
            layer.update_parameter(w_grad[i]*learning_rate,b_grad[i]*learning_rate)

    #resets all gradient related parameters to 0
    def reset(self ):
        for layer in self.layers:
            layer.zero_grad()
            layer.zero_gradsum()

    #returns the gradients for the loss of the previous backward pass
    def get_gradients(self):
        w_grad ,b_grad = [],[]
        for layer in self.layers:
            w_g , b_g = layer.get_grad()
            w_grad.append(w_g)
            b_grad.append(b_g)
        return w_grad, b_grad
    #returns the sum of gradients after the previous reset
    def get_gradientsum(self):
        w_grads, b_grads = [], []
        for layer in self.layers:
            w_g, b_g = layer.get_gradsum()
            w_grads.append(w_g)
            b_grads.append(b_g)
        return w_grads, b_grads