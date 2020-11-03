#!/bin/bash

# Import packages
# %pylab inline
import numpy as np
import sklearn as sk
from sklearn.model_selection import *
import random
import torch
from torch import nn
from torch import optim
from torch.utils.data import random_split, DataLoader


# Eric Jonas: I highly recommend mmapping-in the files -- not necessary for the small one but the others are 20+ GB
ts_spikes = np.load("donkeykong.5000.ts.spikes.npy", mmap_mode='r')
ts_voltage = np.load("donkeykong.5000.ws.spikes.npy", mmap_mode='r')

# Shape: (num_steps x num_transistors)
print("sizes of ts, vs matrices:",ts_spikes.shape, ts_voltage.shape)

data = np.concatenate((ts_spikes,ts_voltage),axis=1)
print('shape of input matrix, shape of a single example:',data.shape, data[1,:].shape)

# Use this to analyze a subset of data
data_subset = data[:100,:]

print("size of subset",data_subset.shape, data_subset[1,:].shape)

data_subset.shape
X = data_subset[0:-1,:]
y = data_subset[1:,:]
y=np.vstack([y, X[len(X)-1,:]])

# Set up device 
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

seed = 1006
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
data_tensor = torch.from_numpy(np.concatenate((X,y),axis=1)).float().to(device)
print('Data Tensor shape:',data_tensor.shape)

# Train, val, test
train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

train_length = int(np.floor(train_ratio*len(data_tensor)))
train_length_2 = int(np.ceil((1-train_ratio)*len(data_tensor)))

train_data_tensor, val_data_tensor = data_tensor[:train_length],data_tensor[train_length_2:]

val_length = int(np.floor(0.5*len(val_data_tensor)))
val_length_2 = int(np.ceil(0.5*len(val_data_tensor)))

val_data_tensor, test_data_tensor = data_tensor[:val_length],data_tensor[val_length_2:]

input_dim = 7020
output_dim = 7020
hidden_dim = 50000
print(input_dim,output_dim,hidden_dim)

# Define model
model = nn.Sequential(
    nn.Linear(input_dim,hidden_dim),
    nn.Sigmoid(),
    nn.Linear(hidden_dim,hidden_dim),
    nn.Sigmoid(),
    nn.Linear(hidden_dim,output_dim),
    nn.Sigmoid()
)

params = list(model.parameters())

# Define optimizer
optimizer = optim.SGD(params, lr=1e-2)

# Define loss
loss = nn.BCELoss()

'''
Note: This is full batch.
'''
def train(model, epochs, x, y,optimizer, criterion):
    
    # Set model to training mode
    model.train()
    
    # Define MSE loss function
    
    for epoch in range(epochs):
        
        losses = list()
        
        y_pred = model(x)

        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        losses.append(loss.item())

        if (epoch+1) % 20 == 0:
            print('Epoch {} loss: {}'.format(epoch+1, torch.tensor(losses).mean() ))
            
        # example of early stopping
        if loss.item() < 1e-2:
            print('Epoch {} loss: {}'.format(epoch+1, torch.tensor(losses).mean() ))
            break
            
    return y_pred.detach()


'''
Note: This is full batch.
'''
def validation(model, x, y, criterion):
    
    model.eval()
            
    losses = list()

    with torch.no_grad():
        y_pred = model(x)

    loss = criterion(y_pred, y)

    losses.append(loss.item())
    print('Validation loss: {}'.format(torch.tensor(losses).mean() ))
            
    return y_pred.detach()

# Training data
x = train_data_tensor[:,:input_dim]
y_true = train_data_tensor[:,input_dim:]
y_pred = train(model, epochs=50, x=x, y=y_true, optimizer=optimizer, criterion = loss)

# # Plot predictions vs actual data
# plt.scatter(x, y_true)
# plt.scatter(x, y_pred)
# plt.show()

# Validation data
x_val = val_data_tensor[:,:input_dim]
y_val_true = val_data_tensor[:,input_dim:]
y_val_pred = validation(model, x=x_val, y=y_val_true, criterion = loss)

# # Plot predictions vs actual data
# plt.scatter(x_val, y_val_true)
# plt.scatter(x_val, y_val_pred)
# plt.show()