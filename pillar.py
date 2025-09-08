from activations import activation_dict,derivative_dict
import jax.numpy as jnp
import time
import jax
import random

'''
BASIC STRUCTURE OF PILLAR CLASS
1) each layer consists of 
    a. Weights and biases
    b. Temporary storage for input , pre activation and activation
    c. activation and activation derivative
2) each layer has the following function
    a. Receiving the output of the previous layers(this is the input for the current layer)
    b. Applying the weights and biases to the input to obtain the preactivation
    c. Applying the activation 
    d. Storing the input ,pre activation and activation temporarily during forward pass
    e. Using the temp data stored during forward pass to calculate the loss gradient
3) management of cache for momentum and adam
    c) i have moved it to a seperate optimizer class
'''
def bernoulli_mask(n, p=0.5):
    # derive a key from current system time
    seed = int(time.time() * 1e6) % (2**31 - 1)
    key = jax.random.PRNGKey(seed)
    rand_vals = jax.random.uniform(key, shape=(n, 1))
    return (rand_vals > p).astype(jnp.int32)

class pillar:
    def __init__(self,input_dim,output_dim,activator = 'relu',beta1 = 0,beta2 = 0,dropout = 0):
        #defining the input and output dimensions for the layer(the input dimension is the output dimension of previous layer
        self.output_dim = output_dim
        self.input_dim = input_dim
        self._size = (input_dim,output_dim)
        #ACTIVATION DATA
        if type(activator) == str:
            self.activator = activation_dict[activator]
            self.activator_derivative = derivative_dict[activator]
        else:
            # If user passed a custom function, fallback to vmap+grad (scalar-safe functions only)
            self.activator = activator
            self.activator_derivative = jax.vmap(jax.grad(self.activator))
        self.dropout = dropout
        #WEIGHTS AND BIASES
        key = jax.random.PRNGKey(random.randint(0,1000))  # random seed
        self.weights = jax.random.normal(key,(self.output_dim,self.input_dim))
        self.bias = jax.random.normal(key,(self.output_dim,1))
        #defining temp cache and setting it to none(forward pass cache)
        self.input = self.pre_activation = self.activation = None
        #defining the gradients for weights and bias (backward pass cache)
        self.weight_grad = jnp.zeros_like(self.weights)
        self.weight_grad_batch_sum = jnp.zeros_like(self.weights)
        self.bias_grad = jnp.zeros_like(self.bias)
        self.bias_grad_batch_sum = jnp.zeros_like(self.bias)


    #returns the size of the layer (aka output_dim,input_dim)
    def size(self):
        return self._size

    def resize(self,input_dim,output_dim):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self._size = (input_dim,output_dim)
        key = jax.random.PRNGKey(0)  # random seed
        self.weights = jax.random.normal(key, (self.output_dim, self.input_dim))
        self.bias = jax.random.normal(key, (self.output_dim, 1))
        # defining temp cache and setting it to none(forward pass cache)
        self.input = self.pre_activation = self.activation = None
        # defining the gradients for weights and bias (backward pass cache)
        self.weight_grad = jnp.zeros_like(self.weights)
        self.weight_grad_batch_sum = jnp.zeros_like(self.weights)
        self.bias_grad = jnp.zeros_like(self.bias)
        self.bias_grad_batch_sum = jnp.zeros_like(self.bias)

    #handles the forward pass (includes the dropout)
    def forward_pass(self,prev_output):
        self.input = prev_output
        self.pre_activation = self.weights @ prev_output + self.bias
        self.activation = self.activator(self.pre_activation)
        return self.activation*bernoulli_mask(self._size[1],p = self.dropout)

    #to be used in forward pass for prediction not training
    def forward(self,prev_output):
        self.input = prev_output
        self.pre_activation = self.weights @ prev_output + self.bias
        self.activation = self.activator(self.pre_activation)
        return self.activation


    #handles the backward pass
    def backward_pass(self,del_l,wl):
        del_curr = self.activator_derivative(self.pre_activation) * (wl.T@del_l)
        self.weight_grad = del_curr @ self.input.T
        self.weight_grad_batch_sum+=self.weight_grad
        self.bias_grad = del_curr
        self.bias_grad_batch_sum += self.bias_grad
        return del_curr , self.weights

    #return functions
    def get_grad(self ):
        return self.weight_grad,self.bias_grad
    def get_gradsum(self):
        return self.weight_grad_batch_sum,self.bias_grad_batch_sum

    #clears cache
    def clear_cache(self):
        self.input = self.pre_activation = self.activation = None

    #clears last run weight and bias gradient
    def zero_grad(self):
        self.weight_grad = jnp.zeros_like(self.weights)
        self.bias_grad = jnp.zeros_like(self.bias)

    #clears the weight grad sum and bias grad sum
    def zero_gradsum(self):
        self.weight_grad_batch_sum = jnp.zeros_like(self.weights)
        self.bias_grad_batch_sum = jnp.zeros_like(self.bias)

    #update functions
    def update_parameter(self,dw,db):
        self.weights -= dw
        self.bias -= db

