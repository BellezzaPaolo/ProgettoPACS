"""@package docstring
Implements Neural Network for Smorzato Test

It is a simple regression problem to approximate the function exp(-x)sin(x). The neural network has 1 input neurons, 1 output neurons, and 
2 hidden layers with 16 neurons each. The activation function is the sigmoid function.
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
torch.manual_seed(12)

class SmorzatoDataset(Dataset):
    """Class inherited from torch.utils.data.Dataset for Smorzato Test
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


class SmorzatoModel(nn.Module):
    '''Class inherited from nn.Module for Smorzato Test
    Args:
        dataset: dataset object of SmorzatoDataset class
        network: neural network architecture
        budget: total number of iterations allowed
        stop_training: boolean variable to stop the training
        criterion: loss function
        opt: chosen optimizer
    '''
    def __init__(self,dataset):
        '''Initialize the SmorzatoModel with the dataset and neural network architecture'''
        super(SmorzatoModel, self).__init__()
        self.network = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1, 16)),
            ('sigmoid1', nn.Sigmoid()),
            ('fc2', nn.Linear(16, 16)),
            ('sigmoid2', nn.Sigmoid()),
            ('fc3', nn.Linear(16, 1)),
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
            data: dictionary containing final training information
        '''

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
                #print('ciao')
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

            # dataset_batched = DataLoader(self.dataset, batch_size=self.dataset.__len__(), shuffle=True)
            
            # # Iterate over batches
            # for batch in dataset_batched:
            #     print(f'Batch size: {batch[0].shape}')
            #     inputs, outputs = batch
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

        if verbose:
            print(f'Training time: {Tend - Tstart:.2f} seconds')
            print(f'Final Loss: {history[-1]:.2e}, final budget: {self.budget} and number of epochs: {epoch}')

        # Save final training data
        data = dict(
            final_loss = history[-1],
            final_budget = self.budget,
            epochs = epoch,
            time_train = Tend - Tstart,
            batch_size = batch_size,
            optimizer_counter = 0
        )
        if hasattr(self.opt, 'counter'):
            data['optimizer_counter'] = self.opt.counter

        return history, data

    def plot_results(self,history):
        '''Plot training loss and decision boundary
        Args:
            history: list of loss values during training
        '''
        # Plot training loss
        k = max(1, len(history)//1000)
        plt.semilogy(history[::k])

        # Plot decision boundary
        x_plot = torch.linspace(0, 10, 1000).reshape(-1, 1)
        y_ex = torch.exp(-x_plot) * torch.sin(x_plot)
        y_test = self.forward(x_plot)

        plt.figure()
        plt.plot(x_plot, y_ex, 'r-', label='Exact function')
        plt.scatter(self.dataset.inputs,self.dataset.outputs, c = 'b', label='Training points')
        plt.plot(x_plot, y_test.detach().numpy(), 'g-', label='NN Prediction')
        plt.show()

    def save_data(self,filename, data, optimizer_name, lr, budget, n_fine=0):
        '''Save training results to a CSV file
        Args:
            filename: name of the CSV file
            data: dictionary containing final training information
            optimizer_name: name of the optimizer used
            lr: learning rate used
            budget: budget used for training
            n_fine: number of fine iterations (default is 0)
        '''
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([optimizer_name, data["batch_size"], lr, data['final_budget'], budget, n_fine, data['final_loss'], data["epochs"], data["time_train"], data['optimizer_counter']])


# Create dataset
x = torch.linspace(0, 10, 30).reshape(-1, 1)

y = torch.exp(-x) * torch.sin(x)

dataset = SmorzatoDataset(x,y)

# Hyperparameters to test
n_fine = [10, 50, 100, 500, 1000, 2000]
learning_r = [1e-1, 1e-2, 1e-3, 1e-4]

budgets = [int(1e4),int(1e5),int(1e6),int(1e7)]
batch_size = int(dataset.__len__())


# Create CSV file to store results and initialize header
filename = "results/Smorzato_results_repr_"+str(batch_size)+".csv"

with open(filename, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['optimizer_name', "batch_size", 'lr', 'final_budget', 'budget', 'n_fine', 'final_loss', "epochs", "time_train", 'optimizer_counter'])

# Run experiments
for lr in learning_r:
    for b in budgets:
        # Train with SGD optimizer
        torch.manual_seed(12)
        model = SmorzatoModel(dataset)
        model.compile(optimizer=torch.optim.SGD(model.parameters(), lr= lr), budget= b)
        history,data = model.train(iterate = b, batch_size = batch_size, display_every= int(b//100), verbose = False, callbacks= [BudgetCallback(b)])
        model.save_data(filename, data, "SGD", lr, b)
        print(f'SGD done for lr: {lr:.2e}, budget: {b:.2e}')

        #model.plot_results(history)

        for nf in n_fine:
            # Train with paraflow optimizer
            torch.manual_seed(12)
            model = SmorzatoModel(dataset)
            model.compile(optimizer=paraflow(model.parameters(), lr_fine=lr, n_fine=nf), budget= b)
            history,data = model.train(iterate = b, batch_size = batch_size, display_every= int(b//100), verbose = False, callbacks= [BudgetCallback(b)])
            model.save_data(filename, data, "paraflow", lr, b, n_fine=nf)
            print(f'paraflow done for lr: {lr:.2e}, budget: {b:.2e}, n_fine:  {nf:.2e}')

            #model.plot_results(history)

