import numpy as np

'''
Basic structure of an optimizer class
1)initialise the optimizer parameters in __init__
2)use morph_over to copy the shape of the layer parameters in the wrapper class
3)in update , get weight and bias gradient terms and use them to update optimizer parameters and return the step direction 
'''

class Adam:
    def __init__(self,beta1 = 0.9,beta2 = 0.99,epsilon = 10e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.w_velocity = []
        self.w_rms = []
        self.b_velocity = []
        self.b_rms = []

        self.time_step = 1

    def morph_over(self,wrapper):
        layers = wrapper.layers if hasattr(wrapper, "layers") else wrapper
        self.w_velocity = [np.zeros_like(layer.weights) for layer in layers]
        self.w_rms = [np.zeros_like(layer.weights) for layer in layers]
        self.b_velocity = [np.zeros_like(layer.bias) for layer in layers]
        self.b_rms = [np.zeros_like(layer.bias) for layer in layers]


    def update(self , grads):
        w_grad, b_grad = grads
        w_step_scaled = []
        b_step_scaled = []
        for i in range(len(w_grad)):
            self.w_velocity[i] = self.w_velocity[i]*self.beta1 + w_grad[i]*(1-self.beta1)
            self.b_velocity[i] = self.b_velocity[i]*self.beta1 + b_grad[i]*(1-self.beta1)
            self.w_rms[i] = self.w_rms[i]*self.beta2 + w_grad[i]**2*(1-self.beta2)
            self.b_rms[i] = self.b_rms[i]*self.beta2 + b_grad[i]**2*(1-self.beta2)

            w_velocity_hat = self.w_velocity[i]/(1.-self.beta1**self.time_step)
            w_rms_hat = self.w_rms[i]/(1.-self.beta2**self.time_step)
            b_velocity_hat = self.b_velocity[i]/(1.-self.beta1**self.time_step)
            b_rms_hat = self.b_rms[i]/(1.-self.beta2**self.time_step)

            w_step_scaled.append(w_velocity_hat/(np.sqrt(w_rms_hat)+self.epsilon))
            b_step_scaled.append(b_velocity_hat/(np.sqrt(b_rms_hat)+self.epsilon))
        self.time_step += 1
        return w_step_scaled, b_step_scaled


class RMSprop:
    def __init__(self, beta=0.9, epsilon=1e-8):
        self.beta = beta
        self.epsilon = epsilon

    def morph_over(self, wrapper):
        layers = wrapper.layers if hasattr(wrapper, "layers") else wrapper
        self.w_rms = [np.zeros_like(layer.weights) for layer in layers]
        self.b_rms = [np.zeros_like(layer.bias) for layer in layers]

    def update(self, grads):
        w_grad, b_grad = grads
        w_step_scaled, b_step_scaled = [], []

        for i in range(len(w_grad)):
            # Update RMS moving average
            self.w_rms[i] = self.beta * self.w_rms[i] + (1 - self.beta) * (w_grad[i] ** 2)
            self.b_rms[i] = self.beta * self.b_rms[i] + (1 - self.beta) * (b_grad[i] ** 2)

            # No bias correction by default (optional)
            w_step_scaled.append(w_grad[i] / (np.sqrt(self.w_rms[i]) + self.epsilon))
            b_step_scaled.append(b_grad[i] / (np.sqrt(self.b_rms[i]) + self.epsilon))


        return w_step_scaled, b_step_scaled


class Momentum:
    def __init__(self, beta=0.9):
        self.beta = beta

    def morph_over(self, wrapper):
        layers = wrapper.layers if hasattr(wrapper, "layers") else wrapper
        self.w_velocity = [np.zeros_like(layer.weights) for layer in layers]
        self.b_velocity = [np.zeros_like(layer.bias) for layer in layers]

    def update(self, grads):
        w_grad, b_grad = grads
        w_step_scaled, b_step_scaled = [], []
        for i in range(len(w_grad)):
            self.w_velocity[i] = self.beta * self.w_velocity[i] + (1 - self.beta) * w_grad[i]
            self.b_velocity[i] = self.beta * self.b_velocity[i] + (1 - self.beta) * b_grad[i]
            w_step_scaled.append(self.w_velocity[i])
            b_step_scaled.append(self.b_velocity[i])
        return w_step_scaled, b_step_scaled


class Nesterov:
    def __init__(self, beta=0.9):
        self.beta = beta

    def morph_over(self, wrapper):
        layers = wrapper.layers if hasattr(wrapper, "layers") else wrapper
        self.w_velocity = [np.zeros_like(layer.weights) for layer in layers]
        self.b_velocity = [np.zeros_like(layer.bias) for layer in layers]

    def update(self, grads):
        w_grad, b_grad = grads
        w_step_scaled, b_step_scaled = [], []
        for i in range(len(w_grad)):
            prev_w = self.w_velocity[i]
            prev_b = self.b_velocity[i]
            self.w_velocity[i] = self.beta * self.w_velocity[i] + w_grad[i]
            self.b_velocity[i] = self.beta * self.b_velocity[i] + b_grad[i]
            w_step_scaled.append((1 + self.beta) * self.w_velocity[i] - self.beta * prev_w)
            b_step_scaled.append((1 + self.beta) * self.b_velocity[i] - self.beta * prev_b)
        return w_step_scaled, b_step_scaled


class Nadam:
    def __init__(self, beta1=0.9, beta2=0.99, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.time_step = 1

    def morph_over(self, wrapper):
        layers = wrapper.layers if hasattr(wrapper, "layers") else wrapper
        self.w_velocity = [np.zeros_like(layer.weights) for layer in layers]
        self.b_velocity = [np.zeros_like(layer.bias) for layer in layers]
        self.w_rms = [np.zeros_like(layer.weights) for layer in layers]
        self.b_rms = [np.zeros_like(layer.bias) for layer in layers]

    def update(self, grads):
        w_grad, b_grad = grads
        w_step_scaled, b_step_scaled = [], []
        for i in range(len(w_grad)):
            self.w_velocity[i] = self.beta1 * self.w_velocity[i] + (1 - self.beta1) * w_grad[i]
            self.b_velocity[i] = self.beta1 * self.b_velocity[i] + (1 - self.beta1) * b_grad[i]
            self.w_rms[i] = self.beta2 * self.w_rms[i] + (1 - self.beta2) * (w_grad[i] ** 2)
            self.b_rms[i] = self.beta2 * self.b_rms[i] + (1 - self.beta2) * (b_grad[i] ** 2)

            w_vel_hat = self.w_velocity[i] / (1 - self.beta1 ** self.time_step)
            b_vel_hat = self.b_velocity[i] / (1 - self.beta1 ** self.time_step)
            w_rms_hat = self.w_rms[i] / (1 - self.beta2 ** self.time_step)
            b_rms_hat = self.b_rms[i] / (1 - self.beta2 ** self.time_step)

            w_nadam = (self.beta1 * w_vel_hat + (1 - self.beta1) * w_grad[i] / (1 - self.beta1 ** self.time_step))
            b_nadam = (self.beta1 * b_vel_hat + (1 - self.beta1) * b_grad[i] / (1 - self.beta1 ** self.time_step))

            w_step_scaled.append(w_nadam / (np.sqrt(w_rms_hat) + self.epsilon))
            b_step_scaled.append(b_nadam / (np.sqrt(b_rms_hat) + self.epsilon))
        self.time_step += 1
        return w_step_scaled, b_step_scaled
