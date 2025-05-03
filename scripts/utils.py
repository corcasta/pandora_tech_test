import os, sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
    
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import torch
import time


def batch_preprocessing(x, y, batch_first=True):
    x_input = x["encoder_cont"]
    y_input = torch.squeeze(y[0])
    if batch_first == False:
        x_input = x_input.permute([1, 0, 2])
    return x_input, y_input


def train_and_validate(
    model, 
    loss_criterion, 
    optimizer, 
    epochs, 
    train_data_loader, 
    valid_data_loader, 
    device,
    save_dir,
    log_dir
):
    writer = SummaryWriter(log_dir=log_dir+"/tensorboard_log")
    history = []
    best_loss = np.inf 
    model = model.to(device)
    
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        epoch_start = time.time()

        # ——— TRAINING ———
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_data_loader, desc="Train", leave=False)
        for inputs, labels in train_pbar:
            x, y = batch_preprocessing(inputs, labels)
            x, y = x.to(device).float(), y.to(device).float()

            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # update tqdm bar
            train_pbar.set_postfix(train_loss=loss.item())

        avg_train_loss = train_loss / len(train_data_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        
        # ——— VALIDATION ———
        model.eval()
        valid_loss = 0.0
        valid_pbar = tqdm(valid_data_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for inputs, labels in valid_pbar:
                x, y = batch_preprocessing(inputs, labels)
                x, y = x.to(device).float(), y.to(device).float()

                outputs = model(x)
                loss = loss_criterion(outputs, y)

                valid_loss += loss.item()
                valid_pbar.set_postfix(valid_loss=loss.item())

        avg_valid_loss = valid_loss / len(valid_data_loader)
        writer.add_scalar("Loss/Validation", avg_valid_loss, epoch)
        
        # ——— LOG & SAVE ———
        elapsed = time.time() - epoch_start
        history.append((avg_train_loss, avg_valid_loss))

        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), save_dir + "/tcn_best.pt")
            print("  → New best model saved")

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Valid Loss: {avg_valid_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )
    writer.close()
    return model, history



"""
def train_and_validate(
    model, 
    loss_criterion, 
    optimizer, 
    epochs, 
    train_data_loader, 
    valid_data_loader, 
    device
    ):
    
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
  
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    
    model = model.to(device)
    start = time.time()
    history = []
    best_loss = np.inf 

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        
        for inputs, labels in tqdm(train_data_loader, desc="Training", leave=False):
            inputs, labels = batch_preprocessing(inputs, labels)
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs.float())

            # Compute loss
            loss = loss_criterion(outputs, labels) #.to(torch.float32)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() #* inputs.size(0)

            # Compute the accuracy
            
        # Set to evaluation mode
        model.eval()
        valid_loss = 0.0,
        with torch.no_grad():
            # Validation loop
            for inputs, labels in tqdm(valid_data_loader, desc=Valid, leave=False):
                inputs, labels = batch_preprocessing(inputs, labels)
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs.cuda().float())

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item()# * inputs.size(0)

        
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/len(train_data_loader) 

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/len(valid_data_loader) 

        history.append([avg_train_loss, avg_valid_loss])        
        epoch_end = time.time()
        
        if avg_valid_loss < best_loss:
            print("New best model saved")
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), 'slippage_model_bw.pt')
            
        print("Epoch : {:03d}, Training: Loss: {:.4f}, \n\t\tValidation : Loss : {:.4f}, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_valid_loss, epoch_end-epoch_start))
  
        # Save if the model has best accuracy till now
        #torch.save(model, dataset+'_model_'+str(epoch)+'.pt')
    #writer.close()      
    return model, history
"""