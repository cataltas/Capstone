#!/bin/bash

# Usage: python -u modeling_v4.py hidden_dim bsize ep LR
# Import packages
import matplotlib.pyplot as plt
import pandas as pd

def load_data(data_path, x_name, y_name):
    import numpy as np
    import torch
    print('loading data')
    X = pd.read_csv(data_path+x_name+'.csv')
    y = pd.read_csv(data_path+y_name+'.csv')
    X = X.sort_values(by  = 'time')
    y = y.sort_values(by = 'time')
    assert np.max(X['time']) == np.max(y['time'])
    X = X.drop(['time'], axis = 1)
    y = y.drop(['time'], axis = 1)
    X_tensor = torch.from_numpy(np.array(X)).float()
    y_tensor = torch.from_numpy(np.array(y)).float()
    assert len(X_tensor) == len(y_tensor)
    assert X_tensor.size()[1] == 1877
    assert y_tensor.size()[1] == 4385
    print("X and y shape:", X_tensor.size(), y_tensor.size())
    print('done loading')

    return X_tensor, y_tensor

# Train, val, test
def train_val_test_split(X_tensor, y_tensor):
    import torch
    train_ratio = 0.6
    validation_ratio = 0.2
    test_ratio = 0.2

    train_length = int(np.floor(train_ratio*len(X_tensor)))

    X_train_data_tensor, X_val_data_tensor = X_tensor[:train_length].float(), X_tensor[train_length:].float()
    y_train_data_tensor, y_val_data_tensor = y_tensor[:train_length].float(), y_tensor[train_length:].float()
    print('X train, val sizes:', X_train_data_tensor.size(), X_val_data_tensor.size())
    print('y train, val sizes:', y_train_data_tensor.size(), y_val_data_tensor.size())

    val_length = int(np.floor(0.5*len(X_val_data_tensor)))

    X_val_data_tensor, X_test_data_tensor = X_tensor[train_length:train_length+val_length].float(), X_tensor[train_length+val_length:train_length+2*val_length].float()
    y_val_data_tensor, y_test_data_tensor = y_tensor[train_length:train_length+val_length].float(), y_tensor[train_length+val_length:train_length+2*val_length].float()
    print('X train, val, test sizes:', X_train_data_tensor.size(), X_val_data_tensor.size(), X_test_data_tensor.size())
    print('y train, val, test sizes:', y_train_data_tensor.size(), y_val_data_tensor.size(), y_test_data_tensor.size())

    print('indices train:', 0, train_length)
    print('indices val:', train_length, train_length+val_length)
    print('indices test:', train_length+val_length, train_length+2*val_length)

    return X_train_data_tensor, y_train_data_tensor, X_val_data_tensor, y_val_data_tensor, X_test_data_tensor, y_test_data_tensor

def set_params(model):
    import torch
    from torch import nn
    from torch import optim

    params = list(model.parameters())

    # Define optimizer
    optimizer = optim.SGD(params, lr=LR)

    # Define loss
    loss_criterion = nn.BCELoss()

    return params, optimizer, loss_criterion

def define_model(input_dim, output_dim, hidden_dim):
    import torch
    from torch import nn
    print('input, output, hidden dim:',input_dim, output_dim, hidden_dim)

    model = nn.Sequential(
        nn.Linear(input_dim,hidden_dim),
        nn.Sigmoid(),
        nn.Linear(hidden_dim,hidden_dim),
        nn.Sigmoid(),
        nn.Linear(hidden_dim,output_dim),
        nn.Sigmoid()
    )
    return model

# Training loop
def train(model, batch_size, epochs, x, y, x_val, y_val, optimizer, criterion):
    import torch
    from torch import nn
    from torch import optim

    print('inside train loop sizes:', x.size(), y.size(), x_val.size(), y_val.size())
    x, y = x.to(device), y.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    model.to(device)
    model.train()

    num_batches = (len(x)+batch_size-1) // batch_size # floor operator
    print('number of batches:',num_batches)
    losslists = []
    vlosslists = []
    for epoch in range(epochs):

        torch.cuda.empty_cache()

        losses = list()
        for b in range(num_batches):

            b_start = b * batch_size
            b_end = (b + 1) * batch_size
            #print('b_start and b_end:', b_start, b_end, flush = True)
            #print('x and y size', x.size(), y.size(), flush =True)
            x_batch = x[b_start:b_end]
            y_batch = y[b_start:b_end]

            torch.cuda.empty_cache()

            y_pred = model(x_batch) # logits
            #print('y pred',y_pred)
            loss = criterion(y_pred, y_batch)
            #print('loss',loss)

            model.zero_grad()

            loss.backward()

            optimizer.step()

            losses.append(loss.item())
            # print(loss.item(), losses)

        losslists.append(np.mean(losses))
        if (epoch+1) % 20 == 0:
            print('Training loss: {}, {}'.format(loss.item(), torch.tensor(losses).mean() ))

        model.eval()

        vlosses = list()

        with torch.no_grad():
            y_pred_val = model(x_val)

            vloss = criterion(y_pred_val, y_val)

            vlosses.append(vloss.item())
        vlosslists.append(torch.tensor(vlosses).mean())

        if (epoch+1) % 20 == 0:
            print('Validation loss: {}, {}'.format(vloss.item(), torch.tensor(vlosses).mean() ))

        if torch.tensor(losses).mean() < 1e-2:
            print('Epoch {} loss: {}, {}'.format(epoch+1, loss.item(), torch.tensor(losses).mean() ))
            break

    return y_pred.detach(), y_pred_val.detach(), losslists, vlosslists

def validation(model, x, y, criterion):

    model.eval()

    losses = list()

    with torch.no_grad():
        y_pred = model(x)
        print('last row of prediction:',y_pred[-1])
        binary_row = (y_pred[-1]>0.5).float().cpu().numpy()
        print('count 1s:', int(np.sum(binary_row)))

    loss = criterion(y_pred, y)

    losses.append(loss.item())
    print('Validation loss: {}, {}'.format(loss.item(), torch.tensor(losses).mean() ))

    return y_pred.detach(), losses

def plot_losses(train_losses, val_losses):
    fig = plt.figure(figsize=(7,7))
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Train & Validation Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.title('Training and Validation Losses')
    plt.savefig('loss_hdim{}_bs{}_ep{}_lr{}.png'.format(hidden_dim,bsize,ep,LR))
    return None

def predict_multiple_steps(X_val_data_tensor, y_val_data_tensor, num_steps, emu_len=152, chip_len=1725):

    import torch
    import numpy as np
    # X and y shape: [9999, 1877] [9999, 4385]
    # input: [152, 1725] EMU & 6507
    # 1725 columns 6507 output, next 2660 are TIA. predicted output size [n,4385], 6507 chip output at t+1 concatenated with TIA chip output
    # predicted output size [n,4385] -> keep only the 6507 chip output
    # Use model to predict next state given previous state's prediction
    n = 1
    # Predict first state
    x_new_6507 = X_val_data_tensor[0:n,emu_len:emu_len+chip_len].float()
    x_new_emu = X_val_data_tensor[0:n,:emu_len].float()
    x_new = np.concatenate((x_new_emu, x_new_6507), axis=1)
    print('shape of new input',x_new.shape)
    x_new = torch.from_numpy(1*(np.array(x_new)>0.5)).float()
    x_new = x_new.to(device)

    # Generate first prediction
    y_new = y_val_data_tensor[0:n,:].float()
    y_new = y_new.to(device)
    print(len(y_new))

    print('len validation (number of steps)',len(x_val))

    # Store generated predictions
    y_gen = list()
    # Store losses
    losses_list = list()
    # Loop over validation set
    for n in range(1,num_steps-1):
        # Predict next state using trained model
        y_next, losses = validation(model, x=x_new, y=y_new, criterion = loss_criterion)
        # print('y_next:', 1*(np.array(y_next)>0.5), np.sum(np.array(y_next)), len(y_next))

        # Update new input to be the prediction of the previous state
        x_new_6507 = y_next[:, :chip_len].float().to(device)
        x_new_emu = X_val_data_tensor[n:n+1, :emu_len].float().to(device)
        print('shapes of emu and 6507',x_new_emu.shape, x_new_6507.shape)
        x_new = torch.cat((x_new_emu, x_new_6507), dim=1)
        print('shape of new input',x_new.shape)
        x_new = x_new.cpu()
        x_new = torch.from_numpy(1*(np.array(x_new)>0.5)).float()
        x_new = x_new.to(device)

        # Update ground truth
        y_new = y_val_data_tensor[n+1,:].float().view(-1, len(y_val_data_tensor[n+1,:]))
        y_new = y_new.to(device)
        # print('y_new:',np.sum(np.array(y_new)), len(y_new))

        losses_list.append(losses)

        # Store generated predictions
        y_gen.append( (y_next>0.5).float().cpu().numpy() )

    print('generate a new input to model based on iterative predictions')
    n = 1
    print('N',n)
    # Store losses
    new_losses_list = list()

    # Predict first state
    # Get first row of EMU
    x_input_emu = X_val_data_tensor[:n, :emu_len].float()
    # Get first row of 6507
    x_input_6507 = X_val_data_tensor[:n, emu_len:emu_len+chip_len].float()
    x_new = np.concatenate((x_input_emu, x_input_6507), axis=1)
    print('shape of first input',x_new.shape)
    x_new = torch.from_numpy(1*(np.array(x_new)>0.5)).float()
    x_new = x_new.to(device)

    # Generate first prediction
    y_new = y_val_data_tensor[:n,:].float()
    y_new = y_new.to(device)
    # print(len(y_new))

    y_gen = list()
    # Loop over validation set
    for n in range(2,num_steps-1):
        print('N',n)
        # Predict next state using trained model
        y_next, losses = validation(model, x=x_new, y=y_new, criterion = loss_criterion)
        # print('y_next:', 1*(np.array(y_next)>0.5), np.sum(np.array(y_next)), len(y_next))
        # y_next has shape [n,4385], first 1725 rows are the 6507 outputs

        # Update new input to be the prediction of the previous state
        x_new_emu = (X_val_data_tensor[ :n, :emu_len]).to(device)
        x_row_1 = (X_val_data_tensor[ :1, emu_len:emu_len+chip_len]).to(device)
        y_next_1 = y_next[ :n, :chip_len].float().to(device)
        x_new_6507 = torch.cat( (x_row_1, y_next_1), axis=0)
        print('shapes of emu and 6507', x_new_emu.shape, x_new_6507.shape)
        x_new = torch.cat((x_new_emu, x_new_6507), dim=1)
        print('shape of new input', x_new.shape)
        x_new = torch.from_numpy(1*(np.array(x_new.cpu())>0.5)).float()
        x_new = x_new.to(device)

        # Update ground truth
        y_new = y_val_data_tensor[:n, :].float()
        y_new = y_new.to(device)
        # print('y_new:',np.sum(np.array(y_new)), len(y_new))

        new_losses_list.append(losses)
        y_gen.append( (y_next>0.5).cpu().float().numpy() )

    print(np.array(losses_list).shape)
    fig = plt.figure(figsize=(20,7))
    plt.plot(np.arange(0,num_steps-1,num_steps/len(losses_list)),losses_list)
    plt.xlabel('n')
    plt.ylabel('BCELoss')
    plt.title('BCELoss predicting next step using previous prediction as input')
    #plt.close()
    plt.savefig('figure2_loss_hdim{}_bs{}_ep{}_lr{}.png'.format(hidden_dim,bsize,ep,LR))

    print(np.array(new_losses_list).shape)
    fig = plt.figure(figsize=(20,7))
    plt.plot(np.arange(0,num_steps-1,num_steps/len(new_losses_list)),new_losses_list)
    plt.xlabel('n')
    plt.ylabel('BCELoss')
    plt.title('BCELoss predicting n successive steps')
    #plt.close()
    plt.savefig('figure3_loss_hdim{}_bs{}_ep{}_lr{}.png'.format(hidden_dim,bsize,ep,LR))

    return None

if __name__ == "__main__":

    import sys
    import numpy as np
    # data_path = '/scratch/fg746/capstone/Capstone/'
    data_path = ''
    hidden_dim = sys.argv[1]
    bsize = sys.argv[2]
    ep = sys.argv[3]
    LR = sys.argv[4]

    hidden_dim = int(hidden_dim)
    bsize = int(bsize)
    ep = int(ep)
    LR = float(LR)
    print( 'hidden dim:', hidden_dim, 'batch size:', bsize, 'epoch:', ep, 'learning rate:', LR)

    X_tensor, y_tensor = load_data(data_path = data_path, x_name = 'X_downsampled', y_name = 'y_downsampled')
    X_train_data_tensor, y_train_data_tensor, X_val_data_tensor, y_val_data_tensor, X_test_data_tensor, y_test_data_tensor = train_val_test_split(X_tensor, y_tensor)

    # Training data
    x = X_train_data_tensor
    y_true = y_train_data_tensor
    print('number of changes in training set:',np.sum(np.array(y_true),axis=1))
    # Validation data
    x_val = X_val_data_tensor
    y_val_true = y_val_data_tensor
    print('number of changes in validation set:',np.sum(np.array(y_val_true),axis=1))

    # Modeling
    import torch
    import random
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

    model = define_model(input_dim = 1877, output_dim = 4385, hidden_dim=hidden_dim)
    params, optimizer, loss_criterion = set_params(model = model)
    y_pred, y_val_pred, train_losses, val_losses = train(model, batch_size=bsize, epochs=ep, x=x, y=y_true, x_val = x_val, y_val = y_val_true, optimizer=optimizer, criterion = loss_criterion)
    torch.cuda.empty_cache()
    plot_losses(train_losses, val_losses)

    predict_multiple_steps(X_val_data_tensor, y_val_data_tensor, 2000)
