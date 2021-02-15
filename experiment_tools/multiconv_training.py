import torch
from torch import nn, optim
import torch.nn.functional as F
import time
from utils import data_loader_wind_us
from models import wind_models
from tqdm import tqdm
import scipy.io as sio
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import load_dataset.reshape_to_batches as reshape_to_batches

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def train_wind_us(dataset, epochs, dev=torch.device("cpu"), patience=None,
                  learning_rate=0.001,
                  kernels_per_layer=16, hidden_neurons=128, batch_size=64):

    print(f"Device: {dev}")


    Xtr, Ytr, Xvalid, Yvalid, Xtest, Ytest = dataset

    input_length = Xtr.shape[1]
    num_output_channel = 1
    num_features = Xtr.shape[-1]
    num_cities = Xtr.shape[-2]
    print("nf")
    print(num_features)

    Xtr, Ytr = reshape_to_batches(Xtr, Ytr, batch_size)
    Xvalid, Yvalid = reshape_to_batches(Xvalid, Yvalid, batch_size)

    print(f'Xtr: {Xtr.shape}')
    print(f'Ytr: {Ytr.shape}')
    print(f'Xvalid: {Xvalid.shape}')
    print(f'Yvalid: {Yvalid.shape}')

    Xtr = torch.as_tensor(Xtr).float()
    Ytr = torch.as_tensor(Ytr).float()
    Xvalid = torch.as_tensor(Xvalid).float()
    Yvalid = torch.as_tensor(Yvalid).float()



    ### Model definition ###
    model = wind_models.MultidimConvNetwork(channels=input_length, height=num_features, width=num_cities,
                                            output_channels=num_output_channel, kernels_per_layer=kernels_per_layer,
                                            hidden_neurons=hidden_neurons)

    # print("Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    summary(model, (input_length, num_cities, num_features), device="cpu")
    # Put the model on GPU
    model.to(dev)
    # Define optimizer
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    # Loss function
    # loss_func = F.mse_loss
    loss_func = F.l1_loss
    #### Training ####
    best_val_loss = 1e300

    early_stopping_epcch = None
    earlystopping_counter = 0
    # pbar = tqdm(range(epochs), desc="Epochs")
    for epoch in range(epochs):
        print(f'epoch: {epoch + 1}/{epochs}')
        model.train()
        train_loss = 0.0
        total_num = 0

        for i, (xb, yb) in enumerate(list(zip(Xtr, Ytr))):
            loss, num = loss_batch(model, loss_func, xb.to(dev), yb.to(dev), opt)
            if loss_func == F.l1_loss:
                num = 1
            train_loss += loss
            total_num += num
        train_loss /= total_num

        # Calc validation loss
        val_loss = 0.0
        val_num = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in list(zip(Xvalid, Yvalid)):

                loss, num = loss_batch(model, loss_func, xb.to(dev), yb.to(dev))
                if loss_func == F.l1_loss:
                    num = 1
                val_loss += loss
                val_num += num
            val_loss /= val_num

        # pbar.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})
        print(f'train_loss: {train_loss}, val_loss: {val_loss}')

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            earlystopping_counter = 0

        else:
            if patience is not None:
                earlystopping_counter += 1
                if earlystopping_counter >= patience:
                    early_stopping_epcch = epoch
                    print(f"Stopping early --> val_loss has not decreased over {patience} epochs")
                    break

    params = [
              ('epochs', epochs),
              ('patience', patience),
              ('early_stopping_epcch', early_stopping_epcch),
              ('learning_rate', learning_rate),
              ('kernels_per_layer', kernels_per_layer),
              ('hidden_neurons', hidden_neurons),
              ('batch_size', batch_size),
    ]
    return model, params