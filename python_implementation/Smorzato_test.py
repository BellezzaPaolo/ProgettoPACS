"""
Implements Neural Network for Smorzato Test:
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
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    

class BudgetCallback(Callback):
    """Set the budget for optimizers that support it.

    Args:
        budget: Total number of iterations allowed.
    """

    def __init__(self, budget):
        super().__init__()
        self.budget = budget

    def on_train_begin(self):
        self.model.budget = self.budget
    
    def on_epoch_end(self):
        if self.model.budget <=0:
            self.model.stop_training = True

            if hasattr(self.model.opt, 'stop_iteration'):
                self.model.opt.stop_iteration = True


class SmorzatoModel(nn.Module):
    def __init__(self,dataset):
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
        return self.network(x)

    def compile(self, optimizer, budget):
        self.budget = budget
        self.opt = optimizer

        if hasattr(self.opt, 'budget'):
            self.opt.budget = budget
        
    def train(self, iterate, batch_size, display_every= 100, verbose=True, callbacks=None):

        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)

        if hasattr(self.opt, 'callbacks'):
            self.opt.callbacks = self.callbacks

        def closure():
            self.budget -= inputs.shape[0]

            if hasattr(self.opt, 'budget'):
                self.opt.budget -= inputs.shape[0]

            y_pred = self.forward(inputs)
            loss = self.criterion(y_pred, outputs)
            self.opt.zero_grad()
            loss.backward()
            return loss
        
        history = []
        Tstart = time.time()

        self.callbacks.on_train_begin()

        for epoch in range(iterate):

            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            dataset_batched = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
            
            for batch in dataset_batched:
                inputs, outputs = batch
                val_loss = self.opt.step(closure)
                if torch.is_tensor(val_loss):
                    val_loss = val_loss.item()

                history.append(val_loss)

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

            # print the behaviour
            if verbose and epoch % display_every == 0:
                print(f'Epoch {epoch}, Loss: {val_loss:.2e} {self.budget}')


        self.callbacks.on_train_end()

        Tend = time.time()

        if verbose:
            print(f'Training time: {Tend - Tstart:.2f} seconds')
            print(f'Final Loss: {history[-1]:.2e}, final budget: {self.budget} and number of epochs: {epoch}')

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
        plt.semilogy(history)

        x_plot = torch.linspace(0, 10, 1000).reshape(-1, 1)
        y_ex = torch.exp(-x_plot) * torch.sin(x_plot)
        y_test = self.forward(x_plot)

        plt.figure()
        plt.plot(x_plot, y_ex, 'r-', label='Exact function')
        plt.scatter(self.dataset.inputs,self.dataset.outputs, c = 'b', label='Training points')
        plt.plot(x_plot, y_test.detach().numpy(), 'g-', label='NN Prediction')
        plt.show()

    def save_data(self,filename, data, optimizer_name, lr, budget, n_fine=0):
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([optimizer_name, data["batch_size"], lr, data['final_budget'], budget, n_fine, data['final_loss'], data["epochs"], data["time_train"], data['optimizer_counter']])


x = torch.linspace(0, 10, 30).reshape(-1, 1)

y = torch.exp(-x) * torch.sin(x)

dataset = SmorzatoDataset(x,y)


n_fine = [10, 50, 100, 500, 1000, 2000]
learning_r = [1e-1, 1e-2, 1e-3, 1e-4]

budgets = [int(1e4),int(1e5),int(1e6),int(1e7)]
batch_size = dataset.__len__()


filename = "results/Smorzato_results.csv"



with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['optimizer_name', "batch_size", 'lr', 'final_budget', 'budget', 'n_fine', 'final_loss', "epochs", "time_train", 'optimizer_counter'])

for lr in learning_r:
    for b in budgets:
        model = SmorzatoModel(dataset)
        model.compile(optimizer=torch.optim.SGD(model.parameters(), lr= lr), budget= b)
        history,data = model.train(iterate = b, batch_size = batch_size, display_every= int(b//100), verbose = False, callbacks= [BudgetCallback(b)])
        model.save_data(filename, data, "SGD", lr, b)
        print('SGD done for lr:', lr, ' budget:', b)

        model.plot_results(history)

        for nf in n_fine:
            model = SmorzatoModel(dataset)

            model.compile(optimizer=paraflow(model.parameters(), lr_fine=lr, n_fine=nf), budget= b)

            history,data = model.train(iterate = b, batch_size = batch_size, display_every= int(b//100), verbose = False, callbacks= [BudgetCallback(b)])
            model.save_data(filename, data, "paraflow", lr, b, n_fine=nf)
            print('paraflow done for lr:', lr, ' budget:', b, ' n_fine:', nf)

            model.plot_results(history)

