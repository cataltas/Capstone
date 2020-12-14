#!/bin/bash
# Usage: python -u modeling_v7.py hidden_dim bsize ep LR

def load_data(X_file, y_file):

    print('loading data')
    X = pd.read_csv(X_file)
    y = pd.read_csv(y_file)
    X = X.sort_values(by  = 'time')
    y = y.sort_values(by = 'time')
    assert np.max(X['time']) == np.max(y['time'])
    X[X.columns[153:]] = X[X.columns[153:]]-1
    y = y-1
    X = X.drop(['time'], axis = 1)
    y = y.drop(['time'], axis = 1)
    # X = X.drop(X.std()[(X.std() == 0)].index, axis=1)
    # y = y.drop(y.std()[(y.std() == 0)].index, axis=1)
    X_tensor = torch.from_numpy(np.array(X)).float()
    y_tensor = torch.from_numpy(np.array(y)).float()
    assert len(X_tensor) == len(y_tensor)

    print("X and y shape:", X_tensor.size(), y_tensor.size())
    print('done loading')

    return X_tensor, y_tensor

# Train, val, test
def train_val_test_split(X_tensor, y_tensor):
    # Define train, val, test split ratio
    train_ratio = 0.6
    validation_ratio = 0.2
    test_ratio = 0.2

    # Make training dataset
    train_length = int(np.floor(train_ratio*len(X_tensor)))

    X_train_data_tensor, X_val_data_tensor = X_tensor[:train_length].float(), X_tensor[train_length:].float()
    y_train_data_tensor, y_val_data_tensor = y_tensor[:train_length].float(), y_tensor[train_length:].float()
    print('X train, val sizes:', X_train_data_tensor.size(), X_val_data_tensor.size())
    print('y train, val sizes:', y_train_data_tensor.size(), y_val_data_tensor.size())

    val_length = int(np.floor(0.5*len(X_val_data_tensor)))

    # Make validation & test datasets
    X_val_data_tensor, X_test_data_tensor = X_tensor[train_length:train_length+val_length].float(), X_tensor[train_length+val_length:train_length+2*val_length].float()
    y_val_data_tensor, y_test_data_tensor = y_tensor[train_length:train_length+val_length].float(), y_tensor[train_length+val_length:train_length+2*val_length].float()
    print('X train, val, test sizes:', X_train_data_tensor.size(), X_val_data_tensor.size(), X_test_data_tensor.size())
    print('y train, val, test sizes:', y_train_data_tensor.size(), y_val_data_tensor.size(), y_test_data_tensor.size())

    print('indices train:', 0, train_length)
    print('indices val:', train_length, train_length+val_length)
    print('indices test:', train_length+val_length, train_length+2*val_length)

    return X_train_data_tensor, y_train_data_tensor, X_val_data_tensor, y_val_data_tensor, X_test_data_tensor, y_test_data_tensor

def set_params(model):

    params = list(model.parameters())

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Define loss
    # TODO: change loss
    loss_criterion =  nn.BCELoss()

    return params, optimizer, loss_criterion

def define_model(input_dim, output_dim, hidden_dim):

    print('input, output, hidden dim:',input_dim, output_dim, hidden_dim)

    # TODO: change activation function
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

    print('inside train loop sizes:', x.size(), y.size(), x_val.size(), y_val.size())
    x, y = x.to(device), y.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    model.to(device)
    model.train()

    num_batches = (len(x)+batch_size-1) // batch_size
    print('number of batches:',num_batches)
    losslists = []
    vlosslists = []
    # Store filename for best model, record how many times best model is found
    filename = ''
    best_model = 0

    for epoch in range(epochs):

        torch.cuda.empty_cache()

        losses = list()
        for b in range(num_batches):

            b_start = b * batch_size
            b_end = (b + 1) * batch_size

            x_batch = x[b_start:b_end]
            y_batch = y[b_start:b_end]

            torch.cuda.empty_cache()

            y_pred = model(x_batch) # logits

            loss = criterion(y_pred, y_batch)

            model.zero_grad()

            loss.backward()

            optimizer.step()

            losses.append(loss.item())

        losslists.append(np.mean(losses))
        print('Epoch {} Training loss: {}, {}'.format(epoch+1, loss.item(), torch.tensor(losses).mean() ))

        # Validation
        model.eval()

        vlosses = list()

        with torch.no_grad():
            y_pred_val = model(x_val)

            vloss = criterion(y_pred_val, y_val)

            vlosses.append(vloss.item())

            # Save the best model
            if epoch == 0:
                print('first epoch loss: {}'.format(vloss.item()))
                best_loss = vloss.item()
            else:
                if vloss.item() < best_loss:
                    print('Best loss: {}, Current loss: {}'.format(best_loss, vloss.item()))
                    best_loss = vloss.item()
                    # Update filename and save best model
                    filename = 'model_hdim{}_bs{}_ep{}_lr{}.pt'.format(hidden_dim, bsize, ep, LR)
                    torch.save(model.state_dict(), filename)
                    best_model += 1

            print('best_model:',best_model)
            # If no better model is found by the end of validation, save current model
            if (epoch == epochs-1 and best_model == 0):
                print('last epoch loss: {}'.format(vloss.item()))
                filename = 'model_hdim{}_bs{}_ep{}_lr{}.pt'.format(hidden_dim, bsize, ep, LR)
                torch.save(model.state_dict(), filename)

            vlosslists.append(torch.tensor(vlosses).mean())

        print('Validation loss: {}, {}'.format(vloss.item(), torch.tensor(vlosses).mean() ))

        if torch.tensor(losses).mean() < 1e-2:
            print('Epoch {} loss: {}, {}'.format(epoch+1, loss.item(), torch.tensor(losses).mean() ))
            break

    print('Val losses:', vlosslists)
    return y_pred.detach(), y_pred_val.detach(), losslists, vlosslists, filename

def validation(model, x, y, criterion):
    # Load the best model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        y_pred = model(x)
        # print(y_pred)

    loss = criterion(y_pred, y)

    # print('Validation loss: {}'.format(loss.item())

    return y_pred.detach(), loss

def test(model_path, x_test, y_test, criterion):
    # Load the best model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x_test, y_test = x_test.to(device), y_test.to(device)

    # Test
    with torch.no_grad():
        y_pred_test = model(x_test)
        tloss = criterion(y_pred_test, y_test)

    return tloss.item()

def plot_losses(train_losses, val_losses):
    fig = plt.figure(figsize=(7,7))
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Train & {} Loss'.format(str))
    plt.legend(['Train', '{}'.format(str)], loc='upper right')
    plt.title('Training and {} Losses'.format(str))
    plt.savefig('loss_hdim{}_bs{}_ep{}_lr{}.png'.format(hidden_dim,bsize,ep,LR))
    return None

def predict_multiple_steps(model_path, X_val_data_tensor, y_val_data_tensor, num_steps, emu_len=152, chip_len=1725):
    '''
    Use model to predict next state given previous state's prediction
    '''

    # Load the best model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    n = 1
    # Predict first state
    x_new = X_val_data_tensor[0:n, :]
    # print('shape of new input',x_new.shape)
    x_new = x_new.to(device)

    # Generate first prediction
    y_new = y_val_data_tensor[0:n,:]
    y_new = y_new.to(device)
    # print(len(y_new))

    # print('len validation (number of steps)',len(x_val))

    # Store generated predictions
    y_gen = list()

    # Store losses
    losses_list = list()

    # Loop over validation set
    for n in range(1,num_steps-1):

        # Predict next state using trained model
        y_next, loss = validation(model, x=x_new, y=y_new, criterion = loss_criterion)
        # print('y_next:', 1*(np.array(y_next)>0.5), np.sum(np.array(y_next)), len(y_next))

        # Update new input to be the prediction of the previous state
        x_new_6507 = y_next[:, :chip_len].float().to(device)
        x_new_6507 = x_new_6507.cpu()
        x_new_6507 = torch.from_numpy(1*(np.array(x_new_6507)>0.5)).float()
        x_new_6507 = x_new_6507.to(device)
        x_new_emu = X_val_data_tensor[n:n+1, :emu_len].float().to(device)
        # print('shapes of emu and 6507',x_new_emu.shape, x_new_6507.shape)
        x_new = torch.cat((x_new_emu, x_new_6507), dim=1)
        # print('shape of new input',x_new.shape)
        x_new = x_new.to(device)

        # Update ground truth
        y_new = y_val_data_tensor[n+1,:].float().view(-1, len(y_val_data_tensor[n+1,:]))
        y_new = y_new.to(device)
        # print('y_new:',np.sum(np.array(y_new)), len(y_new))

        losses_list.append(loss)

        # Store generated predictions
        y_gen.append( (y_next>0.5).float().cpu().numpy() )

    return losses_list

if __name__ == "__main__":
    # Import packages
    import sys
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import torch
    import random
    from torch import nn
    from torch import optim

    # Set parameters
    hidden_dim = sys.argv[1]
    bsize = sys.argv[2]
    ep = sys.argv[3]
    LR = sys.argv[4]

    hidden_dim = int(hidden_dim)
    bsize = int(bsize)
    ep = int(ep)
    LR = float(LR)
    print( 'hidden dim:', hidden_dim, 'batch size:', bsize, 'epochs:', ep, 'learning rate:', LR)

    # Load data
    X_tensor, y_tensor = load_data('X.csv', 'y.csv')
    X_train_data_tensor, y_train_data_tensor, X_val_data_tensor, y_val_data_tensor, X_test_data_tensor, y_test_data_tensor = train_val_test_split(X_tensor, y_tensor)

    # Training data
    x = X_train_data_tensor
    y_true = y_train_data_tensor
    print('number of changes in training set:',np.sum(np.array(y_true),axis=1))

    # Validation data
    x_val = X_val_data_tensor
    y_val_true = y_val_data_tensor
    print('number of changes in validation set:',np.sum(np.array(y_val_true),axis=1))

    # Test data
    x_test = X_test_data_tensor
    y_test_true = y_test_data_tensor
    print('number of changes in test set:',np.sum(np.array(y_test_true),axis=1))

    # Modeling
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
    y_pred, y_val_pred, train_losses, val_losses, best_model_path = train(model, batch_size=bsize, epochs=ep, x=x, y=y_true, x_val = x_val, y_val = y_val_true, optimizer=optimizer, criterion = loss_criterion)
    print('Best model path:', best_model_path)

    # Plot training and validation losses
    plot_losses(train_losses, val_losses)

    # Get test loss on best model
    torch.cuda.empty_cache()
    tloss  = test(model_path = best_model_path, x_test = x_test, y_test = y_test_true, criterion = loss_criterion)
    print('Test loss on best model:', tloss)

    # Iterative predictions
    iterative_losses_list = predict_multiple_steps(model_path = best_model_path, X_val_data_tensor = X_val_data_tensor, y_val_data_tensor = y_val_data_tensor, num_steps=2000)
    # Plot losses from iterative loop
    fig = plt.figure(figsize=(20,7))
    plt.plot(iterative_losses_list)
    plt.xlabel('n')
    plt.ylabel('Loss')
    plt.title('Loss predicting n successive steps')
    plt.savefig('iterative_losses_hdim{}_bs{}_ep{}_lr{}.png'.format(hidden_dim, bsize, ep, LR))
