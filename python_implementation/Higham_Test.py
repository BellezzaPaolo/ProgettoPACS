"""
Implements Neural Network for Higham Test:
It is a simple classification problem with 2d inputs and 2 classes. The neural network has 2 input neurons, 2 output neurons, and 
2 hidden layers with 3 neurons each. The activation function is the sigmoid function.
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

    colors = ['r' if Y[0]==1 else 'b' for Y in y.numpy()]

    X = x.reshape(-1, 2)

    X_plot = torch.linspace(0, 1, 100)
    X1, X2 = torch.meshgrid(X_plot, X_plot)
    z = torch.zeros(X1.shape)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            X_temp = torch.tensor([[X1[i,j], X2[i,j]]], dtype=torch.float32)
            y_temp = model(X_temp)
            if y_temp[0,0] > y_temp[0,1]:
                z[i,j] = 0
            else:
                z[i,j] = 1

    plt.scatter(X[:,0],X[:,1], c = colors)
    plt.contourf(X1, X2, z, alpha=0.3)
    plt.show()



x = torch.tensor([[0.1,0.1],[0.3,0.4],[0.1,0.5],[0.6,0.9],[0.4,0.2],
                  [0.6,0.3],[0.5,0.6],[0.9,0.2],[0.4,0.4],[0.7,0.6]], dtype=torch.float32)

y = torch.tensor([[1,0],[1,0],[1,0],[1,0],[1,0],
                  [0,1],[0,1],[0,1],[0,1],[0,1]], dtype=torch.float32)

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(2, 3)),
    ('sigmoid1', nn.Sigmoid()),
    ('fc2', nn.Linear(3, 3)),
    ('sigmoid2', nn.Sigmoid()),
    ('fc3', nn.Linear(3, 2)),
    ('sigmoid3', nn.Sigmoid())
]))

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
# optimizer = paraflow(model.parameters(), lr_fine=1e-2, n_fine=100)

history = train(model, criterion, optimizer, iterations=1e2)
plot_results(history)
