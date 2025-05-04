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

COLS_ORDER = [
    "Total_Amount", "Age", "Male", "Female", "Quantity",
    "Price_per_Unit", "Year", "Month", "Week",
    "Window_Mean_4", "Window_Mean_5", "Window_Mean_6", "Window_Mean_7"
]

def df_to_tensor(df: pd.DataFrame, products_ids: list) -> torch.Tensor:
    """
    Convert a DataFrame of product time series into a batched PyTorch tensor.
    This function extracts the last 8 time steps of each product’s series from 
    the DataFrame, orders them according to the provided `product_ids` list, 
    and returns a 3D tensor of shape (batch_size, sequence_length, num_features).

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data for 
                           multiple products.
        products_ids (list): List of product IDs specifying which series to 
                             include and in what order.

    Returns:
        torch.Tensor: A float tensor of shape `(len(product_ids), 8, len(COLS_ORDER)), 
                      where:
                    - `len(product_ids)` is the batch size.
                    - `8` is the fixed history window length.
                    - `len(COLS_ORDER)` is the number of features per time step.
    """
        
    # This will only work if all product series are same length
    # WHICH they HAVE TO BE!
    temp_list = []
    for id in products_ids:
        
        df_id = df.loc[df["Product_ID"] == id, COLS_ORDER]
        print(len(df_id))
        temp_list.append(df_id.iloc[-8:,:].to_numpy()[np.newaxis,:,:])
    
    # Each sample in the batch follows same order as product_ids :)
    ts = np.concatenate(temp_list, axis=0).astype(float)
    ts = torch.from_numpy(ts).float()
    return ts
    

def batch_extraction(df: pd.DataFrame, products_ids: list[int]) -> torch.Tensor:
    """
    Preprocess raw transaction data into a batched tensor of time series windows.
    This function transforms and aggregates the input DataFrame, grouping by 
    product category and price, resampling weekly, engineering temporal and 
    rolling features, and finally extracting the last 8 weeks of data per SKU. 
    The result is converted into a 3D tensor via `df_to_tensor`.

    Args:
        df (pd.DataFrame): Raw input DataFrame containing transaction-level data.
        products_ids (list[int]): List of integer product IDs specifying the order 
        in which SKUs should appear in the output tensor. IDs must correspond to 
        the enumeration order of `IDS_MAP.values()`.

    Returns:
        torch.Tensor: A float tensor of shape `(len(product_ids), 8, num_features), 
                      containing the last 8 weeks of engineered features for each 
                      SKU ID.
    """
    
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
    

def batch_preprocessing(x: dict, y: tuple, batch_first: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of encoder input features and target values for model consumption.
    This function extracts the continuous encoder inputs from the feature dict `x`
    and the target tensor from `y`. It optionally permutes the encoder input
    dimensions if the model expects sequence-first format.

    Args:
        x (dict): Dictionary of model inputs. Must contain: "encoder_cont"`
        y (tuple): Tuple containing the target tensor(s). The first element 
                   y[0] should be a tensor of shape (batch_size, ...) 
                   representing the desired labels.
        batch_first (bool, optional): If `True` (default), returns `encoder_cont` 
                                      as-is (batch_size, seq_len, num_features).
                                      If `False`, permutes to 
                                      (seq_len, batch_size, num_features).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The encoder input tensor and 
                                           The squeezed target tensor
    """
    x_input = x["encoder_cont"]
    y_input = torch.squeeze(y[0])
    if batch_first == False:
        x_input = x_input.permute([1, 0, 2])
    return x_input, y_input


def train_and_validate(model: torch.nn.Module, 
                       loss_criterion: callable, 
                       optimizer: torch.optim.Optimizer, 
                       epochs: int, 
                       train_data_loader: torch.utils.data.DataLoader, 
                       valid_data_loader: torch.utils.data.DataLoader, 
                       device: str, 
                       save_dir: str,
                       log_dir: str,
                       model_name: str) -> tuple[torch.nn.Module, list]:
    """
    Train a PyTorch model and validate it each epoch, logging metrics and saving 
    the best model. This function handles the training and validation loops for 
    a given model. It records training and validation losses to TensorBoard, 
    saves the model weights whenever validation loss improves, and returns the 
    trained model along with the loss history.
    
    Args:
        model (torch.nn.Module): The PyTorch model to train.
        loss_criterion (function): Loss function (e.g., torch.nn.MSELoss())
        optimizer (torch.optim.Optimizer): Optimizer configured with model 
                                           parameters (e.g., Adam, SGD)
        epochs (int): Number of training epochs.
        train_data_loader (torch.utils.data.DataLoader): DataLoader for the 
                                                         training set.
        valid_data_loader (torch.utils.data.DataLoader): DataLoader for the 
                                                         validation set.
        device (str): Device on which to perform computation 
                      (e.g., `torch.device("cuda")` or `torch.device("cpu")`).
        save_dir (str): Directory path where the best model checkpoint will 
                        be saved.
        log_dir (str): Directory path for TensorBoard logs.
        model_name (str): Base name for saved model files (will be saved as 
                         `<save_dir>/<model_name>.pt`).

    Returns:
        model (torch.nn.Module): The model after training, moved to the specified 
                                 device.
        history (list): List of `(train_loss, valid_loss)` for each epoch.
    """
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


