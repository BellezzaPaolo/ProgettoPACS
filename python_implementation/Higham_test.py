"""@package docstring
Implements Neural Network for Higham Test

It is a simple classification problem with 2d inputs and 2 classes. The neural network has 2 input neurons, 2 output neurons, and 
2 hidden layers with 3 neurons each. The activation function is the sigmoid function.
"""
import torch
import time
from torch import nn
from collections import OrderedDict
import matplotlib.pyplot as plt
from deepxde.optimizers.pytorch.paraflow import paraflow
from deepxde.callbacks import Callback, CallbackList
from torch.utils.data import Dataset, DataLoader
import csv

class HighamDataset(Dataset):
    """Class inherited from torch.utils.data.Dataset for Higham Test
    Args:
        inputs: inputs data of the nural network
        outputs: real outputs data of the neural network
    """
    def __init__(self, inputs, outputs):
        '''Initialize the dataset with inputs and outputs'''
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        '''Return the total number of samples in the dataset'''
        return len(self.inputs)

    def __getitem__(self, idx):
        '''Retrieve a sample from the dataset at the given index'''
        return self.inputs[idx], self.outputs[idx]

class BudgetCallback(Callback):
    """Class inherited by Callback of deepXDE
    
    Set the budget for optimizers that support it.

    Args:
        budget: Total number of iterations allowed.
    """

    def __init__(self, budget):
        '''Initialize the BudgetCallback with a specified budget'''
        super().__init__()
        self.budget = budget

    def on_train_begin(self):
        '''Set the model's budget at the beginning of training'''
        self.model.budget = self.budget
    
    def on_epoch_end(self):
        '''Check if the budget has been exhausted at the end of each epoch'''
        if self.model.budget <=0:
            self.model.stop_training = True

            if hasattr(self.model.opt, 'stop_iteration'):
                self.model.opt.stop_iteration = True

class HighamModel(nn.Module):
    '''Class inherited from nn.Module for Higham Test
    Args:
        dataset: dataset object of HighamDataset class
        network: neural network architecture
        budget: total number of iterations allowed
        stop_training: boolean variable to stop the training
        criterion: loss function
        opt: chosen optimizer
    '''
    def __init__(self,dataset):
        super(HighamModel, self).__init__()

        # Define the neural network architecture
        self.network = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(2, 3)),
            ('sigmoid1', nn.Sigmoid()),
            ('linear2', nn.Linear(3, 3)),
            ('sigmoid2', nn.Sigmoid()),
            ('linear3', nn.Linear(3, 2)),
            ('sigmoid3', nn.Sigmoid())
        ]))

        self.dataset = dataset
        self.budget = None
        self.stop_training = False
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, x):
        '''Forward pass through the neural network'''
        return self.network(x)

    def compile(self, optimizer, budget):
        '''Compile the model with the chosen optimizer and budget'''
        self.budget = budget
        self.opt = optimizer

        if hasattr(self.opt, 'budget'):
            self.opt.budget = budget
        
    def train(self, iterate, batch_size, display_every= 100, verbose=True, callbacks=None):
        '''Train the neural network
        Args:
            iterate: number of epochs
            batch_size: size of the batches
            display_every: frequency of displaying training progress
            verbose: boolean variable to print training information
            callbacks: list of callback functions
        Returns:
            history: list of loss values during training
            data: dictionary containing final training information'''

        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)

        if hasattr(self.opt, 'callbacks'):
            self.opt.callbacks = self.callbacks
        
        self.batch_size = batch_size

        if hasattr(self.opt, 'batch_fine') and hasattr(self.opt, 'batch_coarse'):
            self.opt.batch_fine = batch_size
            self.opt.batch_coarse = self.dataset.__len__()

        # Define the closure function for optimization
        def closure():
            batch_epoch = None
            if hasattr(self.opt, 'is_coarse'):
                if self.opt.is_coarse:
                    self.budget -= self.opt.batch_coarse
                    batch_epoch = self.opt.batch_coarse
                else:
                    self.budget -= self.opt.batch_fine
                    batch_epoch = self.opt.batch_fine
            else:
                self.budget -= self.batch_size
                batch_epoch = self.batch_size

            if batch_epoch == 1:
                indexes = [torch.randint(self.dataset.__len__() - 1, (1,))]
            elif batch_epoch == self.dataset.__len__():
                indexes = list(range(self.dataset.__len__()))
            else:
                indexes = torch.randperm(self.dataset.__len__())[:batch_epoch]

            y_pred = self.forward(self.dataset.inputs[indexes])
            loss = self.criterion(y_pred, self.dataset.outputs[indexes])
            self.opt.zero_grad()
            loss.backward()
            return loss
        
        history = []
        Tstart = time.time()

        self.callbacks.on_train_begin()

        # Training loop
        for epoch in range(iterate):

            # callbacks at the beginning of epoch and batch
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            # train step
            val_loss = self.opt.step(closure)
            if torch.is_tensor(val_loss):
                val_loss = val_loss.item()

            history.append(val_loss)

            # callbacks at the end of batch and epoch
            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

            # print the behaviour
            if verbose and epoch % display_every == 0:
                y_pred = self.forward(self.dataset.inputs)
                Loss = self.criterion(y_pred, self.dataset.outputs).item()
                print(f'Epoch {epoch}, Loss: {Loss:.2e} {self.budget}')


        self.callbacks.on_train_end()

        Tend = time.time()

        # evaluate the loss at the end of training
        y_pred = self.forward(self.dataset.inputs)
        loss = self.criterion(y_pred, self.dataset.outputs).item()

        if verbose:
            print(f'Training time: {Tend - Tstart:.2f} seconds')
            print(f'Final Loss: {loss:.2e}, final budget: {self.budget} and number of epochs: {epoch}')

        # Save final training data
        data = dict(
            final_loss = loss,
            final_budget = self.budget,
            epochs = epoch,
            time_train = Tend - Tstart,
            batch_size = batch_size,
            optimizer_counter = 0
        )
        if hasattr(self.opt, 'counter'):
            data['optimizer_counter'] = self.opt.counter

        return history, data

    def plot_results(self, history):
        '''Plot training loss and decision boundary
        Args:
            history: list of loss values during training
        '''
        # Plot training loss
        k = max(1, len(history)//1000)
        plt.semilogy(history[::k])

        # Plot decision boundary
        colors = ['r' if Y[0]==1 else 'b' for Y in self.dataset.outputs.numpy()]

        X = self.dataset.inputs.reshape(-1, 2)

        X_plot = torch.linspace(0, 1, 100)
        X1, X2 = torch.meshgrid(X_plot, X_plot)
        z = torch.zeros(X1.shape)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                X_temp = torch.tensor([[X1[i,j], X2[i,j]]], dtype=torch.float32)
                y_temp = self.forward(X_temp)
                if y_temp[0,0] > y_temp[0,1]:
                    z[i,j] = 0
                else:
                    z[i,j] = 1

        plt.figure()
        plt.scatter(X[:,0],X[:,1], c = colors)
        plt.contourf(X1, X2, z, alpha=0.3)
        plt.show()

    def save_data(self,filename, data, optimizer_name, lr, budget, n_fine=0):
        '''Save training results to a CSV file
        Args:
            filename: name of the CSV file
            data: dictionary containing final training information
            optimizer_name: name of the optimizer used
            lr: learning rate used
            budget: budget used for training
            n_fine: number of fine iterations (default is 0)'''
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([optimizer_name, data["batch_size"], lr, data['final_budget'], budget, n_fine, data['final_loss'], data["epochs"], data["time_train"], data['optimizer_counter']])

# Define the dataset
x = torch.tensor([[0.1,0.1],[0.3,0.4],[0.1,0.5],[0.6,0.9],[0.4,0.2],
                  [0.6,0.3],[0.5,0.6],[0.9,0.2],[0.4,0.4],[0.7,0.6]], dtype=torch.float32)

y = torch.tensor([[1,0],[1,0],[1,0],[1,0],[1,0],
                  [0,1],[0,1],[0,1],[0,1],[0,1]], dtype=torch.float32)

dataset = HighamDataset(x, y)


# example to run the code once not all the test

# lr = 1e-2
# b= int(1e4)
# nf = 100
# model = HighamModel(dataset)
# model.compile(optimizer=torch.optim.SGD(model.parameters(), lr= lr), budget= b)
# #model.compile(optimizer=paraflow(model.parameters(), lr_fine=lr, n_fine=nf), budget= b)
# history,data = model.train(iterate = b, batch_size = batch_size, display_every= 1000, verbose = True, callbacks= [BudgetCallback(b)])
# print(f'SGD done for lr: {lr:.2e},  budget: {b:.2e}')

# model.plot_results(history)

# Test the code on different learning rate and budgets

# Training parameters settings
n_fine = [10, 50, 100, 500, 1000, 2000]
learning_r = [1e-1, 1e-2, 1e-3, 1e-4]

budgets = [int(1e4),int(1e5),int(1e6),int(1e7)]
batch_size = int(dataset.__len__()/2)


# Create results file and write header
filename = "results/Higham_results_repr_"+str(batch_size)+".csv"

with open(filename, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['optimizer_name', "batch_size", 'lr', 'final_budget', 'budget', 'n_fine', 'final_loss', "epochs", "time_train", 'optimizer_counter'])

# Run experiments
for lr in learning_r:
    for b in budgets:
        # Train with SGD optimizer
        torch.manual_seed(12)
        model = HighamModel(dataset)
        model.compile(optimizer=torch.optim.SGD(model.parameters(), lr= lr), budget= b)
        history,data = model.train(iterate = b, batch_size = batch_size, display_every= int(b//100), verbose = False, callbacks= [BudgetCallback(b)])
        model.save_data(filename, data, "SGD", lr, b)
        print(f'SGD done for lr: {lr:.2e},  budget: {b:.2e}')

        # model.plot_results(history)

        for nf in n_fine:
            # Train with paraflow optimizer
            torch.manual_seed(12)
            model = HighamModel(dataset)
            model.compile(optimizer=paraflow(model.parameters(), lr_fine=lr, n_fine=nf), budget= b)
            history,data = model.train(iterate = b, batch_size = batch_size, display_every= int(b//100), verbose = False, callbacks= [BudgetCallback(b)])
            model.save_data(filename, data, "paraflow", lr, b, n_fine=nf)
            print(f'paraflow done for lr: {lr:.2e}, budget: {b:.2e}, n_fine: {nf}')

            # model.plot_results(history)

