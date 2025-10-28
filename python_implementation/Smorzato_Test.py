"""
Implements Neural Network for Smorzato Test:
It is a simple regression problem to approximate the function exp(-x)sin(x). The neural network has 1 input neurons, 1 output neurons, and 
2 hidden layers with 16 neurons each. The activation function is the sigmoid function.
"""
                   
import torch
from torch import nn
from collections import OrderedDict
import matplotlib.pyplot as plt
from deepxde.optimizers.pytorch.paraflow import paraflow

torch.manual_seed(12)


def train(model, criterion, optimizer, iterations,verbose=True):
    def closure():
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        return loss
    
    history = []
    for t in range(int(iterations)):
        
        val_loss = optimizer.step(closure)
        if torch.is_tensor(val_loss):
            val_loss = val_loss.item()

        history.append(val_loss)

        # print the behaviour
        if t % 1000 == 0 and verbose:
            print(f'Iteration {t}, Loss: {val_loss:.2e}')
    return history

def plot_results(history):
    plt.semilogy(history)
    plt.show()

    x_plot = torch.linspace(0, 10, 1000).reshape(-1, 1)
    y_ex = torch.exp(-x_plot) * torch.sin(x_plot)
    y_test = model(x_plot)
    plt.plot(x_plot, y_ex, 'r-', label='Exact function')
    plt.scatter(x,y, c = 'b', label='Training points')
    plt.plot(x_plot, y_test.detach().numpy(), 'g-', label='NN Prediction')
    plt.show()


x = torch.linspace(0, 10, 30).reshape(-1, 1)

y = torch.exp(-x) * torch.sin(x)

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1, 16)),
    ('sigmoid1', nn.Sigmoid()),
    ('fc2', nn.Linear(16, 16)),
    ('sigmoid2', nn.Sigmoid()),
    ('fc3', nn.Linear(16, 1)),
    ('sigmoid3', nn.Sigmoid())
]))

criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
optimizer = paraflow(model.parameters(), lr_fine=1e-2, n_fine=100)

history = train(model, criterion, optimizer, iterations=1e2)
plot_results(history)