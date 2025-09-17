from loss import loss_dict , loss_derivative_dict
import pillarMC
import downhill
import scheduler
import pillar
import numpy as np
import random

#converts any list type data into the correct shape
def rectifier(data, n):
    try:
        arr = np.array(data)
    except Exception:
        raise TypeError("Data could not be converted to a NumPy array.")
    if arr.size != n:
        raise ValueError(f"Data must have {n} elements, but it has {arr.size}.")

    return arr.reshape(n, 1)

class piron:
    def __init__(self,loss = 'mse'):
        #defining loss function and its derivative
        self.loss = loss_dict[loss]
        self.loss_derivative = loss_derivative_dict[loss]
        #defining wrapper object and optimizer
        self.wrapper = pillarMC.pillarMC()
        self.optimizer = None
        #defining data
        self.data = None
        #defining scheduler class
        self.scheduler = None

    #adds layer to the model
    def add(self,layer,morph = False):
        self.wrapper.add(layer,morph)

    #loads data ino the model class
    def load_data(self,data):
        self.data = data
    #appends additional data to existing data
    def append_data(self,extra_data):
        for data_point in extra_data:
            self.data.append(data_point)
    #rectifies the data into the correct shape
    def rectify_data(self):
        indim,outdim = self.wrapper.inout_dim()
        rectified_data = []
        for x,y in self.data:
            rectified_data.append((rectifier(x,indim),rectifier(y,outdim)))
        self.data = rectified_data
    #loads the optimizer
    def load_optimizer(self,optimizer):
        self.optimizer = optimizer
        self.optimizer.morph_over(self.wrapper)
    #loads the scheduler
    def load_scheduler(self,scheduler):
        self.scheduler = scheduler

    #one complete epoch run
    def epoch_run(self,batch_size = None,lr = 10e-3):
        l = len(self.data)
        if l ==0 :
            raise Exception('Data set empty : please use load_data()')
        if self.optimizer is None:
            raise Exception('No optimizer defined. Please use load_optimizer()')
        if self.scheduler is None:
            raise Exception('No scheduler defined. Please use load_scheduler()')
        random.shuffle(self.data)

        if batch_size is None:
            batch_size = l
        q,r = l//batch_size,l%batch_size

        epoch_loss = 0

        for i in range(q):
            sample = self.data[i*batch_size:(i+1)*batch_size]
            for data_point in sample:
                x,y = data_point
                y_pred = self.wrapper.fwd_pass(x)
                epoch_loss += np.sum(self.loss(y_pred,y))
                loss_grad = self.loss_derivative(y_pred,y)
                self.wrapper.bkwd_pass(loss_grad)

            grads = self.wrapper.get_gradientsum()
            steps = self.optimizer.update(grads)
            self.wrapper.update_parameters(steps,learning_rate=lr)

        tail_sample = self.data[l-r:]
        for data_point in tail_sample:
            x, y = data_point
            y_pred = self.wrapper.fwd_pass(x)
            epoch_loss += np.sum(self.loss(y_pred, y))
            loss_grad = self.loss_derivative(y_pred, y)
            self.wrapper.bkwd_pass(loss_grad)

        grads = self.wrapper.get_gradientsum()
        steps = self.optimizer.update(grads)
        self.wrapper.update_parameters(steps, learning_rate=lr)
        self.wrapper.reset()
        return epoch_loss
    #
    def train(self,batch_size = None,epochs = 100,tolerance = 0):
        loss_history = []
        for i in range(epochs):
            learning_rate = self.scheduler.get_lr()
            loss = self.epoch_run(batch_size = batch_size,lr = learning_rate)
            loss_history.append(loss)
            print(f"\rEpoch: {i+1}|Loss: {loss}",end='', flush=True)
            if loss < tolerance:
                break
        print('')
        return loss_history

    def reset_scheduler(self):
        self.scheduler.reset()

    def output(self,x):
        indim,_ = self.wrapper.inout_dim()
        return self.wrapper.output(rectifier(x,indim))


