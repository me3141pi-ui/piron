import jax.numpy as jnp
import math
class exp_decay:
    def __init__(self, initial_lr = 10e-3 ,k = 1):
        self.lr = initial_lr
        self.k = k
        self.epoch = 0
    def get_lr(self):
        self.epoch += 1
        return self.lr * math.exp(-self.k * self.epoch)
    def reset(self):
        self.epoch = 0
class step_decay:
    def __init__(self, initial_lr = 10e-3 ,k = 1,step_size = 30):
        self.lr = initial_lr
        self.k = k
        self.step_size = step_size
        self.epoch = 0
    def get_lr(self):
        self.epoch += 1
        return self.lr * math.exp(-self.k * self.epoch//self.step_size)
    def reset(self):
        self.epoch = 0