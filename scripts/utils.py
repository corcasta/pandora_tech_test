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
    log_dir,
    model_name
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
            torch.save(model.state_dict(), save_dir + f"/{model_name}.pt")
            print("  → New best model saved")

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Valid Loss: {avg_valid_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )
    writer.close()
    return model, history


