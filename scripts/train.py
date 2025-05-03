import os, sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
    
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting import TimeSeriesDataSet
from scripts.utils import train_and_validate
from models.tcn import TCNPredictor
from torch import nn, optim
from pathlib import Path
import pandas as pd
import torch

# MODEL PARAMS
MIN_ENCODER_LENGTH    = 8
MAX_ENCODER_LENGTH    = 8
MAX_PREDICTION_LENGTH = 4

# TRAINING PARAMS
BATCH_SIZE = 4
EPOCHS = 400
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA PARAMS
DATASET_PATH = str(Path(os.getcwd()).parent) + "/data"


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


    train_sample_x, train_sample_y = next(iter(train_dataloader))

    
    model = TCNPredictor(input_size=len(input_features)+1, 
                         seq_len=MAX_ENCODER_LENGTH, 
                         output_size=MAX_PREDICTION_LENGTH).to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    save_model_path = proj_root + "/models/weights"
    trained_model, history = train_and_validate(
        model, 
        loss_fn, 
        optimizer, 
        EPOCHS, 
        train_dataloader, 
        valid_dataloader, 
        DEVICE,
        save_model_path
    )

if __name__ == "__main__":
    main()