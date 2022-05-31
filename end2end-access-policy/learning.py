import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader, random_split

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import copy

####################
# Dataset
####################
# Set random seed
torch.manual_seed(0)

# Define batch size
batch_size = 512

# Load dataset
dataset_full = np.load('dataset.npz')

x_full = dataset_full['x']#.transpose(0, 2, 1)
y_full = dataset_full['y']

# Use just the magnitude
x_ = torch.from_numpy(np.abs(x_full)).float().cuda()
y_ = torch.from_numpy(y_full).cuda()

# Split dataset
x_train = x_[:7000]
y_train = y_[:7000]

x_val = x_[7000:8500]
y_val = y_[7000:8500]

x_test = x_[8500:]
y_test = y_[8500:]

# Data standardization
mean = x_train.mean()
std = x_train.std()

x_train = (x_train - mean)/std
x_val = (x_val - mean)/std
x_test = (x_test - mean)/std

# Dataset
train_dataset = TensorDataset(x_train, y_train)
validation_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Extract input dimensions
in_channels = torch.tensor(x_.shape[1:])

# Extract output dimensions
out_channels = len(torch.unique(y_))

####################
# Model
####################


class Feedfoward(nn.Module):

    def __init__(self, in_channels, out_channels):

        # Paretizing
        super(Feedfoward, self).__init__()

        # Get dimensions
        self.in_dim = int(torch.prod(in_channels))
        self.out_dim = out_channels

        self.neurons = ((self.in_dim + self.out_dim) // 2)

        # Define layers
        self.hidden_layer_1 = nn.Linear(self.in_dim, self.neurons).cuda()
        self.hidden_layer_2 = nn.Linear(self.neurons, self.neurons).cuda()

        self.output_layer = nn.Linear(self.neurons, self.out_dim).cuda()

    def forward(self, x):
        out = x.reshape(x.shape[0], self.in_dim)

        out = F.relu(self.hidden_layer_1(out))
        out = F.relu(self.hidden_layer_2(out))

        out = self.output_layer(out)

        return out

    def loss(self, y_pred, y_true):
        return F.cross_entropy(input=y_pred, target=y_true)

    def __str__(self):
        return "FF"


class CNN(nn.Module):

    def __init__(self, in_channels, out_channels):

        # Paretizing
        super(CNN, self).__init__()

        # Get dimensions
        self.in_dim = in_channels
        self.out_dim = out_channels

        self.neurons = ((torch.prod(in_channels).int() + self.out_dim) // 2)

        # CNN parameters
        channels = 64
        kernel = 3
        stride = 1

        cnn_out_dim_1 = int(((in_channels[0] - kernel) / stride) + 1)
        cnn_out_dim_2 = int(((in_channels[0] - kernel) / stride) + 1)

        # Define layers
        self.hidden_layer_1 = nn.Conv2d(1, channels, kernel, stride=1).cuda()
        self.hidden_layer_2 = nn.Linear((channels * (cnn_out_dim_1//2) * (cnn_out_dim_2//2)), self.neurons, bias=True).cuda()

        self.output_layer = nn.Linear(self.neurons, self.out_dim, bias=True).cuda()

    def forward(self, x):
        out = x[:, None, :, :]

        out = F.relu(self.hidden_layer_1(out))
        out = F.max_pool2d(out, 2, stride=2)
        out = out.flatten(start_dim=1)
        out = F.relu(self.hidden_layer_2(out))

        out = self.output_layer(out)

        return out

    def loss(self, y_pred, y_true):
        return F.cross_entropy(input=y_pred, target=y_true)

    def __str__(self):
        return "CNN"


class RNN(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim):

        # Paretizing
        super(RNN, self).__init__()

        # Store dimensions
        self.in_dim = in_channels[-1]
        self.hidden_dim = hidden_dim
        self.out_dim = out_channels

        self.neurons = ((torch.prod(in_channels).int() + self.out_dim) // 2)

        # Define layers
        self.hidden_layer_1 = nn.GRU(self.in_dim, hidden_dim, batch_first=True).cuda()
        self.hidden_layer_2 = nn.Linear(in_channels[0]*hidden_dim, self.neurons, bias=True).cuda()

        self.output_layer = nn.Linear(self.neurons, self.out_dim, bias=True).cuda()

    def forward(self, x):

        # Initialize hidden state for first input with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).cuda()

        out, _ = self.hidden_layer_1(x, h0)
        out = out.flatten(start_dim=1)
        out = F.relu(self.hidden_layer_2(out))

        out = self.output_layer(out)

        return out

    def loss(self, y_pred, y_true):
        return F.cross_entropy(input=y_pred, target=y_true)

    def __str__(self):
        return "RNN"


# Function to evaluate Model Complexity
def get_no_params(model):
    layer_params = {}

    for ll, param in enumerate(model.parameters()):
        layer_params[ll] = []

        for s in list(param.size()):
            layer_params[ll].append(s)

    return layer_params

####################
# Training and Validation
####################

# Define learning rate
learning_rate = 0.01

# Define momentum
momentum = 0.9

# Define weight decay
weight_decay = 1e-6

# Define optimizers
optimizers = ['sgd', 'adam']

# Number of epochs
n_epochs = 50

# Prepare to save results
results = {'sgd': {}, 'adam': {}}
bestmodel = {'sgd': {'model':None, 'test':1.}, 'adam': {'model':None, 'test':1.}}

# Go through all different optimizers
for optz in optimizers:

    # Instantiate the model
    model = Feedfoward(in_channels, out_channels)
    model = CNN(in_channels, out_channels)
    model = RNN(in_channels, out_channels, hidden_dim=32)

    # Model Complexity
    results['complexity'] = get_no_params(model)

    # Build the optimizer
    if optz == 'sgd':
        opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                        weight_decay=weight_decay)
    elif optz == 'adam':
        opt = optim.Adam(model.parameters())  # default parameters
    else:
        break

    # Prepare to save training results
    results[optz]['train'] = {'loss': [], 'acc': []}
    results[optz]['validation'] = {'loss': [], 'acc': []}
    results[optz]['test'] = {'acc': []}

    # Go through all epochs
    for epoch in tqdm(range(n_epochs)):

        ###############
        # Training
        ###############

        # Go through all batches
        for train_batch, (x, y) in enumerate(train_dataloader):

            model.train()

            opt.zero_grad()

            y_pred = model(x)

            loss = model.loss(y_pred, y)

            loss.backward()
            opt.step()

            with torch.no_grad():
                acc = y_pred.softmax(dim=1).argmax(dim=1).eq(y).sum().item() / len(y)

            results[optz]['train']['loss'].append(loss.item())
            results[optz]['train']['acc'].append(acc)

            if train_batch % 10 == 0 or (train_batch + 1) == len(train_dataloader):
                print(f'epoch: {epoch}, train batch index: {train_batch}, training loss: {loss:>6f}, training acc: {acc:>2f}')

            ###############
            # Validation
            ###############

            # Create total validation loss
            val_loss = 0.0
            val_correct = 0

            for val_batch, (x, y) in enumerate(validation_dataloader):
                model.eval()

                with torch.no_grad():
                    y_pred = model(x)
                    #y_pred = y_pred_softmax.argmax(dim=-1)

                    loss = model.loss(y_pred, y)

                    with torch.no_grad():
                        correct = y_pred.softmax(dim=1).argmax(dim=1).eq(y).sum().item()

                    val_loss += loss.item()
                    val_correct += correct

            results[optz]['validation']['loss'].append(val_loss / len(validation_dataloader))
            results[optz]['validation']['acc'].append(val_correct / len(validation_dataset))

        ###############
        # Testing
        ###############
        model.eval()

        y_pred = model(x_test)

        with torch.no_grad():
            acc = y_pred.softmax(dim=1).argmax(dim=1).eq(y_test).sum().item() / len(y_test)

        results[optz]['test']['acc'].append(acc)

    print("##########################")
    print("model: ", model)
    print("##########################")

    print("----- Test Accuracy ------")

    print(f"{100 * results[optz]['test']['acc'][-1]:2f}")

    if bestmodel[optz]['test'] >= acc:
        bestmodel[optz]['test'] = acc
        bestmodel[optz]['model'] = copy.deepcopy(model)

    #del model

print("----- Model Complexity ------")
print(results['complexity'])

####################
# Plots
####################
fig, ax = plt.subplots()

styles = ['-', ':']

# Go through all different optimizers
for oo, optz in enumerate(optimizers):

    ax.plot(range(len(results[optz]['train']['loss'])), results[optz]['train']['loss'], linestyle=styles[oo], label=(optz + ': train'))
    ax.plot(range(len(results[optz]['validation']['loss'])), results[optz]['validation']['loss'], linestyle=styles[oo], label=(optz +': validation'))

    plt.gca().set_prop_cycle(None)

ax.set_xlabel('training iteration')
ax.set_ylabel('loss')

ax.set_yscale('log')

ax.legend(framealpha=0.5)
ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

# plt.show()

#####

fig, ax = plt.subplots()

styles = ['-', ':']

# Go through all different optimizers
for oo, optz in enumerate(optimizers):

    ax.plot(range(len(results[optz]['train']['acc'])), results[optz]['train']['acc'], linestyle=styles[oo], label=(optz + ': train'))
    ax.plot(range(len(results[optz]['validation']['acc'])), results[optz]['validation']['acc'], linestyle=styles[oo], label=(optz +': validation'))

    plt.gca().set_prop_cycle(None)

ax.set_xlabel('training iteration')
ax.set_ylabel('accuracy')

ax.legend(framealpha=0.5)
ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

# plt.show()

#####

fig, ax = plt.subplots()

styles = ['-', ':']

# Go through all different optimizers
for oo, optz in enumerate(optimizers):

    ax.plot(range(len(results[optz]['test']['acc'])), results[optz]['test']['acc'], linestyle=styles[oo], label=optz)

    plt.gca().set_prop_cycle(None)

ax.set_xlabel('epoch')
ax.set_ylabel('test accuracy')

if n_epochs == 50:
    ax.set_xticks([0, 10, 20, 30, 40, 50])

ax.legend(framealpha=0.5)
ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

# plt.show()

####################
# Save
####################

# Save results
np.savez(str(model),
         sgd=results['sgd']['test']['acc'],
         adam=results['adam']['test']['acc']
)





