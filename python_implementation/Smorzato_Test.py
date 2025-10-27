"""
Implements Neural Network for Smorzato Test:
It is a simple regression problem to approximate the function exp(-x)sin(x). The neural network has 1 input neurons, 1 output neurons, and 
2 hidden layers with 16 neurons each. The activation function is the sigmoid function.
"""
                   
import torch
from torch import nn
from collections import OrderedDict
import matplotlib.pyplot as plt

torch.manual_seed(12)

x = torch.linspace(0, 10, 30)

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
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

history = []
for t in range(int(1e6)):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute loss
    loss = criterion(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    history.append(loss.item())

    # print the behaviour
    if t % 1000 == 0:
        print(f'Iteration {t}, Loss: {loss.item():.2e}')

plt.semilogy(history)
plt.show()

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

plt.scatter(X[:,0],X[:,1])
plt.contourf(X1, X2, z, alpha=0.3)
plt.show()