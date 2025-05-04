#import os, sys
#PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#if PROJECT_ROOT not in sys.path:
#    sys.path.insert(0, PROJECT_ROOT)

from torch.utils.tensorboard import SummaryWriter
from config import PROJECT_ROOT
from torch import nn, optim
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import time

IDS_MAP = {
    ('Beauty', 25): 0,
    ('Beauty', 30): 1,
    ('Beauty', 50): 2,
    ('Beauty', 300): 3,
    ('Beauty', 500): 4,
    ('Clothing', 25): 5,
    ('Clothing', 30): 6,
    ('Clothing', 50): 7,
    ('Clothing', 300): 8,
    ('Clothing', 500): 9,
    ('Electronics', 25): 10,
    ('Electronics', 30): 11,
    ('Electronics', 50): 12,
    ('Electronics', 300): 13,
    ('Electronics', 500): 14
 }

PRODUCT_IDS = [i for i in range(len(IDS_MAP))]

COLS_ORDER = [
    "Total_Amount", "Age", "Male", "Female", "Quantity",
    "Price_per_Unit", "Year", "Month", "Week",
    "Window_Mean_4", "Window_Mean_5", "Window_Mean_6", "Window_Mean_7"
]

def df_to_tensor(df: pd.DataFrame, products_ids: list):
    # This will only work if all product series are same length
    # WHICH they HAVE TO BE!
    temp_list = []
    for id in products_ids:
        
        df_id = df.loc[df["Product_ID"] == id, COLS_ORDER]
        temp_list.append(df_id.iloc[-8:,:].to_numpy()[np.newaxis,:,:])
    
    # Each sample in the batch follows same order as product_ids :)
    ts = np.concatenate(temp_list, axis=0).astype(float)
    ts = torch.from_numpy(ts).float()
    return ts
    

def batch_extraction(df: pd.DataFrame, products_ids: list[int]):
    df["Date"] = pd.DatetimeIndex(df["Date"])
    # Renaming Columns
    new_column_names = []
    for name in df.columns:
        new_column_names.append(name.replace(" ", "_"))    
    df.columns = new_column_names

    for gender in ["Male", "Female"]:
        df[gender] = (df["Gender"] == gender).astype(int)
            
    df["Product_ID"] = 0
    df_prods_list = []
    agg_dict = {
        "Age": "median",
        "Male": "sum",
        "Female": "sum",
        "Quantity": "sum",
        "Total_Amount": "sum"
    }
    
    for id, (category, price) in enumerate(IDS_MAP.keys()):
        df_product = df.loc[(df["Product_Category"] == category) & (df["Price_per_Unit"] == price)].sort_values(by=["Date"])
        df_product = df_product.groupby([pd.Grouper(key="Date", freq="W-Mon")]).agg(agg_dict).reset_index()
        
        df_product["Product_ID"] = id
        df_product["Product_Category"] = category
        df_product["Price_per_Unit"] = price
        df_product = df_product.fillna(0)
        
        df_product["Time_Unitless"] = [i for i in range(len(df_product))]
        df_product["Year"] = df_product["Date"].dt.year
        df_product["Month"] = df_product["Date"].dt.month
        df_product["Week"] = df_product["Date"].dt.isocalendar().week
        df_product.pop("Date")
        
        individual_windows = 4
        for window in range(4, individual_windows+4):
            df_product[f"Window_Mean_{window}"] = df_product["Total_Amount"].rolling(window, ).mean().bfill().ffill().round(2)
            #window_columns.append(f"Window_Mean_{window}")
        df_prods_list.append(df_product.iloc[-8:])
    
    df_final = pd.concat(df_prods_list, ignore_index=True)    
    return df_to_tensor(df_final, products_ids)
    

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


