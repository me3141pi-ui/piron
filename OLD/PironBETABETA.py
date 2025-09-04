import numpy as np
import pickle
import random
import jax.numpy as jnp
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


###ACTIVATION FUNCTION DERIVATIVES
def identity_derivative(x):
    return 1

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def signum_derivative(x):
    return 0 if x!=0 else 1

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

def prelu_derivative(x, alpha=0.25):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def softplus_derivative(x):
    return sigmoid(x)

def softsign_derivative(x):
    return 1 / (1 + np.abs(x))**2

def swish_derivative(x, beta=1.0):
    s = sigmoid(beta * x)
    return s + beta * x * s * (1 - s)

def gelu_derivative(x):
    tanh_term = np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * (x**3)))
    return 0.5 * (1 + tanh_term) + 0.5 * x * (1 - tanh_term**2) * \
           (np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * (x**2)))

###SINGLE COMPONENT LOSS FUNCTIONS
def rms_loss(y_true, y_pred):
    return (y_true-y_pred)**2
def cross_entropy_loss(y_true, y_pred):
    return -y_true * np.log(y_pred)

###SINGLE COMPONENT LOSS DERIVATIVE
def rms_loss_derivative(y_true, y_pred):
    return 2*(y_pred - y_true)
def cross_entropy_loss_derivative(y_true, y_pred):
    return -y_true/y_pred

### MATRIX RELATED FUNCTIONS
#transpose_1d returns the nx1 dimensional array of a single dimensional np array of length n
def transpose_1d(arr):
    return arr.reshape(len(arr), 1)

# DEFINING THE CLASS PIRON FOR THE NEURAL NETWORK BODY
#BASIC STRUCTURE :
#NEURAL NETWORK ACCEPTS EITHER AN NX1 2 DIMENSIONAL NP ARRAY OR A SINGLE DIMENSIONAL ARRAY OF LENGTH N
#WHILE THE SINGLE DIMENSIONAL ARRAY IS ACCEPTED AS AN INPUT , THE INTERNALS WORK SOLELY ON TREATING THE INPUT AS A COLUMN VECTOR
#SAME FOR OUTPUT
class piron:
    def __init__(self, layer_info, activation_function=0, output_activation=1, loss=0):
        #DEFINING THE ACTIVATION FUNCTION FOR NON OUTPUT LAYERS (EXCEPT FOR THE FIRST ONE) and THE OUTPUT FUNCTION
        activators = [relu, sigmoid, signum , identity]
        self.activator = activators[activation_function]
        self.output_activation = activators[output_activation]
        activator_derivatives = [relu_derivative, sigmoid_derivative, signum_derivative,identity_derivative]
        self.activator_derivative = activator_derivatives[activation_function]
        self.output_derivative = activator_derivatives[output_activation]

        #DEFINING THE LOSS FUNCTION
        loss_functions = [rms_loss, cross_entropy_loss]
        self.loss_function = loss_functions[loss]
        loss_derivatives = [rms_loss_derivative, cross_entropy_loss_derivative]
        self.loss_derivative = loss_derivatives[loss]

        #DEFINING THE LAYER INFO
        self.layer_info = layer_info
        self.n_layers = len(layer_info) # counts the input (dormant) layer as well
        #CREATE THE WEIGHT MATRICES FOR EACH LAYER AND THEIR RESPECTIVE BIAS MATRICES
        # O O O O O     0th layer (first layer has no activation function)
        #  0 0 0 0      1st layer (1st layer , its weight matrix W~[1] is of the dimension 4 x 5)(W~[n] = weight matrix for nth layer)
        #   0 0 0       2nd layer
        #    0 0        3rd layer
        #     0         4th layer
        #FOR THE nth LAYER THE INPUT IS A COLUMN VECTOR OF THE DIMENSION length(nth layer) x 1
        #FOR THE nth LAYER(n>0) THE WEIGHT MATRIX IS OF THE DIMENSIONS length(nth layer) x length(n-1th layer)
        #THE BIAS MATRIX IS A COLUMN VECTOR OF THE DIMENSION length(nth layer) x 1
        self.weight_matrix = [np.random.randn(self.layer_info[i + 1], t) for i, t in enumerate(layer_info[0:-1])]
        self.bias_matrix = [np.random.randn(t, 1) for t in layer_info[1:]]

    #RECTIFIER FUNCTION WHICH CORRECTS X (FOR INPUT CASE) and Y (FOR OUTPUT CASE)
    def rectifier(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        n = self.layer_info[0]
        if x.ndim == 1 and x.shape[0] == n:
            return x.reshape(-1, 1)
        elif x.ndim == 2 and x.shape == (n, 1):
            return x
        elif x.ndim == 2 and x.shape == (1, n):
            return x.reshape(-1, 1)
        else:
            raise ValueError(f"Input must be a vector of shape ({n},), ({n},1), or (1,{n})")

    def rectifier2(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        n = self.layer_info[-1]
        if x.ndim == 1 and x.shape[0] == n:
            return x.reshape(-1, 1)
        elif x.ndim == 2 and x.shape == (n, 1):
            return x
        elif x.ndim == 2 and x.shape == (1, n):
            return x.reshape(-1, 1)
        else:
            raise ValueError(f"Input must be a vector of shape ({n},), ({n},1), or (1,{n})")

    ###RETURNS THE ACTIVATION VECTOR(AKA OUTPUT) FOR EACH LAYER IN THE NEURAL NETWORK
    def activation_and_input_matrix(self, x):
        z = self.rectifier(x)
        activation_list = [z]
        inputs = [z]
        for M,C in zip(self.weight_matrix[:-1], self.bias_matrix[:-1]):
            z = M@z + C
            inputs.append(z)
            z = self.activator(z)
            activation_list.append(z)
        Mf,Cf = self.weight_matrix[-1],self.bias_matrix[-1]
        z= Mf@z + Cf
        inputs.append(z)
        z = self.output_activation(z)
        activation_list.append(z)
        return activation_list, inputs


    # RETURNS THE OUTPUT OF THE NEURAL NETWORK
    def neural_output(self, input_vector):
        z = self.rectifier(input_vector)
        for M,C in zip(self.weight_matrix[:-1], self.bias_matrix[:-1]):
            z = self.activator(M@z + C)

        Mf, Cf = self.weight_matrix[-1], self.bias_matrix[-1]
        z = self.output_activation(Mf @ z + Cf)
        return z

    #LOADS DATA INTO THE OBJECT
    def load_data(self,data):
        self.training_data = []
        for x,y in data:
            self.training_data.append((self.rectifier(x), self.rectifier2(y)))

    def save_data(self,filename):
        with open(filename,'wb') as f:
            pickle.dump(self.training_data,f)


    #SINGLE POINT GRADIENT DATA LOSS
    #THE MOST COMPLICATED PART OF THE PROJECT
    #CALCULATES THE GRADIENT VECTOR FOR ALL THE WEIGHTS AND BIASES
    #THE GRADIENT IS WRT TO THE LOSS FUNCTION FOR A SINGLE DATA POINT
    def single_data_point_gradient(self,data_point):
        x,y = data_point
        activation,inputs = self.activation_and_input_matrix(x)
        del_temp = self.output_derivative(inputs[-1])*self.loss_derivative(y,activation[-1])
        bias_gradient = [del_temp]
        weight_gradient = [del_temp@activation[-2].T]
        for i in range(self.n_layers-2,0,-1):
            del_temp = self.activator_derivative(inputs[i])*(self.weight_matrix[i].T@del_temp)
            weight_gradient.append(del_temp@activation[i-1].T)
            bias_gradient.append(del_temp)

        return (weight_gradient[::-1], bias_gradient[::-1])
    def rand_grad(self):
        return self.single_data_point_gradient(random.choice(self.training_data))
    def update_weight(self  ,step ,learning_constant = 10**(-3)):
        for i in range(self.n_layers-1):
            self.weight_matrix[i] -= step[i]*learning_constant
    def update_bias(self,step,learning_constant=10**(-3)):
        for i in range(self.n_layers-1):
            self.bias_matrix[i] -= step[i]*learning_constant

    def update_parameters(self, wg, bg, learning_constant=1e-3):
        for i in range(self.n_layers - 1):
            self.weight_matrix[i] -= learning_constant * wg[i]
            self.bias_matrix[i] -= learning_constant * bg[i]


    def save_model(self, filename):
        model_data = {
            'weights': self.weight_matrix,
            'biases': self.bias_matrix,
            'layer_info': self.layer_info
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

    @staticmethod
    def load_model(filename, activation_function=0, output_activation=1, loss=0):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)

        # Create a new object with matching layer info
        model = piron(
            layer_info=model_data['layer_info'],
            activation_function=activation_function,
            output_activation=output_activation,
            loss=loss
        )

        # Override the weights and biases
        model.weight_matrix = model_data['weights']
        model.bias_matrix = model_data['biases']
        return model


