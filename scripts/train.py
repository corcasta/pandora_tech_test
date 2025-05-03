import os, sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
    
    
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting import TimeSeriesDataSet
import torch
from torch import nn, optim
from pathlib import Path
import pandas as pd
from models.tcn import TCNPredictor

from scripts.utils import train_and_validate

# MODEL PARAMS
MIN_ENCODER_LENGTH    = 4
MAX_ENCODER_LENGTH    = 8
MAX_PREDICTION_LENGTH = 4

# TRAINING PARAMS
BATCH_SIZE = 4
EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA PARAMS
DATASET_PATH = str(Path(os.getcwd()).parent) + "/data"


def batch_preprocessing(x, y, batch_first=True):
    x_input = x["encoder_cont"]
    y_input = torch.squeeze(y[0])
    if batch_first == False:
        x_input = x_input.permute([1, 0, 2])
    return x_input, y_input


def main():
    train_df = pd.read_csv(DATASET_PATH + "/train_data.csv")
    valid_df = pd.read_csv(DATASET_PATH + "/valid_data.csv")
    
    input_features = list(train_df.columns.drop(["Product_ID", "Total_Amount", "Product_Category", "Time_Unitless"]))
    feature_scalers = [None]*len(input_features)
    scalers_dict = dict(zip(input_features, feature_scalers))
    
    # Dataset definitiona: Train & Valid
    train_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="Time_Unitless",
        target="Total_Amount",
        target_normalizer=None,
        categorical_encoders={"Product_Category": NaNLabelEncoder().fit(train_df.Product_Category)},
        group_ids=["Product_ID"],
        static_categoricals=["Product_Category"],
        time_varying_unknown_reals=["Total_Amount", *input_features],
        min_encoder_length=MIN_ENCODER_LENGTH,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        scalers=scalers_dict,
    )
    
    valid_dataset = TimeSeriesDataSet(
        valid_df,
        time_idx="Time_Unitless",
        target="Total_Amount",
        target_normalizer=None,
        categorical_encoders={"Product_Category": NaNLabelEncoder().fit(valid_df.Product_Category)},
        group_ids=["Product_ID"],
        static_categoricals=["Product_Category"],
        time_varying_unknown_reals=["Total_Amount", *input_features],
        min_encoder_length=MIN_ENCODER_LENGTH,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        scalers=scalers_dict,
    ) 
    
    # Dataloaders definition: Train & Valid
    train_dataloader = train_dataset.to_dataloader(batch_size=BATCH_SIZE)
    valid_dataloader = valid_dataset.to_dataloader(batch_size=BATCH_SIZE)

    model = TCNPredictor(input_size=len(input_features)+1, 
                         seq_len=MAX_ENCODER_LENGTH, 
                         output_size=MAX_PREDICTION_LENGTH).to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trained_model, history = train_and_validate(
        model, 
        loss_fn, 
        optimizer, 
        EPOCHS, 
        train_dataloader, 
        valid_dataloader, 
        DEVICE
    )

if __name__ == "__main__":
    main()