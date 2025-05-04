    
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting import TimeSeriesDataSet
from scripts.utils import train_and_validate
from models.tcn import TCNPredictor
from config import PROJECT_ROOT
from torch import nn, optim
from config import DEVICE
import pandas as pd
#import torch
import time
import json
import os

# MODEL PARAMS
MIN_ENCODER_LENGTH    = 8
MAX_ENCODER_LENGTH    = 8
MAX_PREDICTION_LENGTH = 4

# TRAINING PARAMS
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 0.001
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA PARAMS
DATASET_PATH = PROJECT_ROOT + "/data"

def log_training_state(log_dir: str, train_df: pd.DataFrame, valid_df: pd.DataFrame, train_dataset: TimeSeriesDataSet):
    td = train_dataset.get_parameters()
    info = {
        "train_params":{
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
        "model_params":{
            "min_encoder_length": MIN_ENCODER_LENGTH,
            "max_encoder_length": MAX_ENCODER_LENGTH,
            "max_prediction_length": MAX_PREDICTION_LENGTH,
        },
        "train_dataframe": dict(
            zip(train_df.dtypes.reset_index()["index"], train_df.dtypes.reset_index()[0].astype("str"))
        ),
        "valid_dataframe": dict(
            zip(valid_df.dtypes.reset_index()["index"], valid_df.dtypes.reset_index()[0].astype("str"))
        ),
        "time_varying_unknown_reals": td["time_varying_unknown_reals"]
    }
    
    with open(log_dir+'/train_state.json', 'w') as f:
        json.dump(info, f, indent=4)


def main():
    # Logs folder creation
    named_tuple = time.localtime()
    time_string = time.strftime("%Y_%m_%d_%H_%M_%S", named_tuple)
    log_dir = PROJECT_ROOT+f"/logs/train_logs/run_{time_string}" 
    os.mkdir(log_dir)
    os.mkdir(log_dir+"/tensorboard_log")
    
    # Load datasets
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
    
    # Create log dirs
    log_training_state(log_dir, train_df, valid_df, train_dataset)
    
    
    # Dataloaders definition: Train & Valid
    train_dataloader = train_dataset.to_dataloader(batch_size=BATCH_SIZE)
    valid_dataloader = valid_dataset.to_dataloader(batch_size=BATCH_SIZE)

    #train_sample_x, train_sample_y = next(iter(train_dataloader))

    model = TCNPredictor(input_size=len(input_features)+1, 
                         seq_len=MAX_ENCODER_LENGTH, 
                         output_size=MAX_PREDICTION_LENGTH).to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    save_model_dir = PROJECT_ROOT + "/models/weights"
    trained_model, history = train_and_validate(
        model, 
        loss_fn, 
        optimizer, 
        EPOCHS, 
        train_dataloader, 
        valid_dataloader, 
        DEVICE,
        save_model_dir,
        log_dir,
        model_name=f"tcn_{time_string}"
    )

if __name__ == "__main__":
    main()