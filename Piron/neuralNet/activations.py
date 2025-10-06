import numpy as np

###ACTIVATION FUNCTIONS
def identity(x):
    return x

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def signum(x):
    return x/(x**2)**0.5 if x!=0 else 0

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def prelu(x, alpha=0.25):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softplus(x):
    return np.log1p(np.exp(x))

def softsign(x):
    return x / (1 + np.abs(x))

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * (x**3))))



# ACTIVATION dictionary (functions only)
activation_dict = {
    "identity": identity,
    "relu": relu,
    "sigmoid": sigmoid,
    "signum": signum,
    "tanh": tanh,
    "leaky_relu": leaky_relu,
    "prelu": prelu,
    "elu": elu,
    "softplus": softplus,
    "softsign": softsign,
    "swish": swish,
    "gelu": gelu,
}
derivative_dict = {
    "identity": lambda x: 1,
    "relu": lambda x: (x > 0).astype(x.dtype),
    "sigmoid": lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))),
    "tanh": lambda x: 1 - np.tanh(x)**2,
    "leaky_relu": lambda x, alpha=0.01: np.where(x > 0, 1, alpha),
    "prelu": lambda x, alpha=0.25: np.where(x > 0, 1, alpha),
    "elu": lambda x, alpha=1.0: np.where(x > 0, 1, alpha * np.exp(x)),
    "softplus": lambda x: 1 / (1 + np.exp(-x)),
    "softsign": lambda x: 1 / (1 + np.abs(x))**2,
    "swish": lambda x, beta=1.0: (1 / (1 + np.exp(-beta * x))) + x * (beta * (1 / (1 + np.exp(-beta * x))) * (1 - 1 / (1 + np.exp(-beta * x)))),
    "gelu": lambda x: 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))) + 0.5 * x * (1 - np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))**2) * np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * x**2)
}

def get_activations():
    print('Activation Functions:')
    for key in activation_dict.keys():
        print(key)



