#!/bin/bash

# Import packages
import numpy as np
import random
import torch
from torch import nn
from torch import optim
from torch.utils.data import random_split
# from sklearn.utils import resample
import matplotlib.pyplot as plt

print('loading data')
# Eric Jonas: I highly recommend mmapping-in the files -- not necessary for the small one but the others are 20+ GB
ts_spikes = np.load("donkeykong.5000.ts.spikes.npy", mmap_mode='r')
ts_voltage = np.load("donkeykong.5000.ws.spikes.npy", mmap_mode='r')
print('done loading')

# Shape: (num_steps x num_transistors)
data = np.concatenate((ts_spikes,ts_voltage),axis=1)
print('shape of input matrix:',data.shape)

# Use this to analyze a subset of data
subset_size = 10000
data_subset = data[:subset_size,:]
print("size of subset:",data_subset.shape)

X = data_subset
y = data_subset[1:,:]
y = np.vstack([y, X[len(X)-1,:]])

# Set up device
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
print('device:',device)

seed = 1006
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

data_tensor = torch.from_numpy(np.concatenate((X,y),axis=1)).float()
print('Data Tensor shape:',data_tensor.shape)

# Train, val, test
train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

train_length = int(np.floor(train_ratio*len(data_tensor)))
train_length_2 = int(np.ceil((1-train_ratio)*len(data_tensor)))

train_data_tensor, val_data_tensor = data_tensor[:train_length].float(),data_tensor[train_length_2:].float()

val_length = int(np.floor(0.5*len(val_data_tensor)))
val_length_2 = int(np.ceil(0.5*len(val_data_tensor)))

val_data_tensor, test_data_tensor = data_tensor[:val_length].float(),data_tensor[val_length_2:].float()

input_dim = 7020
output_dim = 7020
hidden_dim = 50000
print('input, output, hidden dim:',input_dim,output_dim,hidden_dim)

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
loss_criterion = nn.BCELoss()

# Training loop
def train(model, batch_size, epochs, x, y,optimizer, criterion):

    # Set model to training mode
    model.to(device)
    model.train()

    # Define MSE loss function
    num_batches = (len(x)+batch_size-1) // batch_size # floor operator
    print('number of batches:',num_batches)

    for epoch in range(epochs):

        torch.cuda.empty_cache()

        losses = list()
        for b in range(num_batches):

            b_start = b * batch_size
            b_end = (b + 1) * batch_size

            x_batch = x[b_start:b_end]
            y_batch = y[b_start:b_end]

            torch.cuda.empty_cache()
            #print('losses',losses)

            y_pred = model(x_batch) # logits
            #print('y pred',y_pred)

            loss = criterion(y_pred, y_batch)
            #print('loss',loss)

            model.zero_grad()

            loss.backward()

            optimizer.step()

            losses.append(loss.item())
            # print(loss.item(), losses)

        if (epoch+1) % 20 == 0:
            print('Epoch {} loss: {}, {}'.format(epoch+1, loss.item(), torch.tensor(losses).mean() ))

        if loss.item() < 1e-2:
            print('Epoch {} loss: {}, {}'.format(epoch+1, loss.item(), torch.tensor(losses).mean() ))
            break

    return y_pred.detach(), losses

def validation(model, x, y, criterion):

    model.eval()

    losses = list()

    with torch.no_grad():
        y_pred = model(x)

    loss = criterion(y_pred, y)

    losses.append(loss.item())
    print('Validation loss: {}, {}'.format(loss.item(), torch.tensor(losses).mean() ))

    return y_pred.detach(), losses

# Training data
x = train_data_tensor[:,:input_dim].float().to(device)
y_true = train_data_tensor[:,input_dim:].float().to(device)

print('number of changes in training set:',np.sum(np.array(y_true),axis=1))

# Set number of epochs
ep = 100
bsize = int(subset_size//5)
print('batch size',bsize)

#print('epochs:',ep)
#print('len x:',len(x)/5, int(len(x)/5))

y_pred, losses = train(model, batch_size=bsize, epochs=ep, x=x, y=y_true, optimizer=optimizer, criterion = loss_criterion)
train_loss = losses
torch.cuda.empty_cache()

# Validation data
x_val = val_data_tensor[:,:input_dim].float().to(device)
y_val_true = val_data_tensor[:,input_dim:].float().to(device)
y_val_pred,losses = validation(model, x=x_val, y=y_val_true, criterion = loss_criterion)
val_loss = losses

print('number of changes in validation set:',np.sum(np.array(y_val_true),axis=1))

fig = plt.figure(figsize=(7,7))
plt.plot(np.arange(0,100,100/len(train_loss)),train_loss)
#plt.close()
fig.savefig('figure.pdf')

# Use model to predict next state given previous state's prediction
n = 1
# Predict first state
x_new = val_data_tensor[0:n,:input_dim].float().to(device)
print(len(x_new))
# Generate first prediction
y_new = val_data_tensor[0:n,input_dim:].float().to(device)
print(len(y_new))

print('len validation',len(x_val))

# Store generated predictions
y_gen = list()
# Store losses
losses_list = list()
# Loop over validation set
for n in range(1,len(x_val)):
    # Predict next state using trained model
    y_next, losses = validation(model, x=x_new, y=y_new, criterion = loss_criterion)
    # print('y_next:', 1*(np.array(y_next)>0.5), np.sum(np.array(y_next)), len(y_next))

    # Update new input to be the prediction of the previous state
    x_new = torch.from_numpy(1*(np.array(y_next)>0.5)).float().to(device)

    # Update ground truth
    y_new = val_data_tensor[n-1:n,input_dim:].float().to(device)
    # print('y_new:',np.sum(np.array(y_new)), len(y_new))

    losses_list.append(losses)

    # Store generated predictions
    y_gen.append( (y_next>0.5).float().numpy() )

print('generate a new input to model based on iterative predictions')

# Convert predictions to binary
gen_data = torch.from_numpy(1*(np.array(y_gen)>0.5)).float().to(device)
#print(gen_data)
#print(np.sum(1*(np.array(y_gen)>0.5),0))

#print(gen_data.shape)
# Reshape [n x d]
gen_data = gen_data.view((len(y_gen),input_dim),-1)
print('reshaped:',gen_data.shape)

# Store losses
losses_new = list()
print('validate input made up of n successive states')
# Loop over number of successive steps
for n in range(1,len(x_val)-1):
    x_new = gen_data[0:n].float().to(device)
    y_new = val_data_tensor[0:n,input_dim:].float().to(device)
    # Predict next state using trained model
    y_next, losses = validation(model, x=x_new, y=y_new, criterion = loss_criterion)
    # print('y_next:', 1*(np.array(y_next)>0.5), np.sum(np.array(y_next)), len(y_next))
    losses_new.append(losses)

fig = plt.figure(figsize=(7,7))
plt.plot(np.arange(0,100,100/len(losses_new)),losses_new)
#plt.close()
fig.savefig('figure2.pdf')
