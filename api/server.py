from config import (PROJECT_ROOT, DEVICE, 
                    MODEL_WEIGHTS, NUM_FEATURES, 
                    SEQ_LEN, OUT_LEN)
from scripts.utils import batch_extraction
from fastapi import HTTPException, status
from models.tcn import TCNPredictor
import litserve as ls 
import pandas as pd
import numpy as np
import torch
import json
import time

class SimpleLogger(ls.Logger):
    def process(self, key: str, value: float):
        """
        Log each metric received by printing to stdout.

        Args:
            key (str): The name of the metric (e.g., "model_time")
            value (float): The value of the metric.
        """
        print(f"Received {key} with value {value}", flush=True)

class FileLogger(ls.Logger):
    def process(self, key: str, value: float):
        """
         Append each metric received to a file in the logs directory.

        Args:
            key (str): The name of the metric (e.g., "model_time").
            value (float): The value of the metric.
        """
        with open(PROJECT_ROOT+"/logs/api_logs/litserve_metrics.log", "a+") as f:
            f.write(f"{key}: {value:.1f}\n")
       
class ForecastAPI(ls.LitAPI):
    def setup(self, device: str):
        """
        Load the model and prepare it for inference on the specified device.

        Args:
            device (str): Device identifier, e.g. "cpu" or "cuda"
        """
        self.device = device
        self.model = TCNPredictor(NUM_FEATURES, SEQ_LEN, OUT_LEN).to(device)
        self.model.load_state_dict(torch.load(MODEL_WEIGHTS, weights_only=True, map_location=device))
        self.model.eval()
    
    def decode_request(self, request, **kwargs) -> torch.Tensor:
        """
        Validate and decode an incoming HTTP request into a model-ready tensor.

        Args:
            request (FormData): Dictionary representing multipart form data. Must contain:
                            - "metadata": JSON string with key "product_id" (list of ints).
                            - "file": Uploaded CSV file of raw transaction data.

        Raises:
            HTTPException
            400 if "metadata" or "file" keys are missing.
            422 if metadata is malformed, missing required keys, or no valid IDs.
            422 if CSV parsing fails.

        Returns:
            torch.Tensor: Batched time-series tensor of shape (batch_size, seq_len, num_features).
        """
        if "metadata" not in request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing metadata key in form-data upload."
            )
        
        if "file" not in request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing csv file in form-data upload."
            )
        
        metadata = json.loads(request["metadata"])
        if "product_id" not in metadata.keys():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="product_id key is required inside metadata."
            )

        if len(metadata["product_id"]) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="product_id content must have at least 1 element."
            )
            
        self.product_ids = []
        self.product_ids_invalid = [] 
        
        for pid in metadata["product_id"]:
            pid = int(pid)
            # For this demo we can do this simple checkup.
            if pid < 0 or pid > 14:
                self.product_ids_invalid.append(pid)
            else:
                self.product_ids.append(pid)      
        
        if len(self.product_ids) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Not a single valid product_id was given. Provide at least 1 (values from 0-14)."
            )
  
        if "file" not in request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing CSV file in form-data upload"
            )
        
        # Dataframe extraction
        try:
            csv = request["file"].file
            df = pd.read_csv(csv)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Error parsing CSV: {e}"
            )
        
        # Each sample in the batch follows same order as self.product_ids :)
        batch = batch_extraction(df, self.product_ids)
        print(batch.shape)
        if tuple(batch.shape[1:]) != (8, 13):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not enough consecutive weeks, it requires minimum 8."
            )
        return batch
    
    def predict(self, x: torch.Tensor, **kwargs) -> np.ndarray:
        """
        Perform inference on a batch of input tensors and log the model execution time.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            np.ndarray: Model output tensor of shape (batch_size, out_len).
        """
        t0 = time.time()
        prediction = self.model.predict(x.to(self.device))
        t1 = time.time()
        self.log("model_time", t1 - t0)
        return prediction
    
    def encode_response(self, output: np.ndarray, **kwargs) -> dict:
        """
        Convert model output array into a JSON-serializable dict.
        
        Args:
            output (np.ndarray): Model output of shape (batch_size, out_len).

        Returns:
            dict:  Mapping from product ID (as string) to a dict of week forecasts.
        """
        if self.device != "cpu":
            output = output.cpu().numpy()
        else:
            output = output.numpy()
        response = {}
        for idx, product_id in enumerate(self.product_ids):
            week_data = {}
            for week in range(output.shape[1]):
                week_data[f"week_{week}"] = str(output[idx, week])
            response[str(product_id)] = week_data
        return response

if __name__ == "__main__":
    api = ForecastAPI()
    server = ls.LitServer(api, accelerator=DEVICE, loggers=[SimpleLogger(), FileLogger()])
    server.run(port=8000, generate_client_file=False)