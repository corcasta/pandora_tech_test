from config import (PROJECT_ROOT, DEVICE, 
                    MODEL_WEIGHTS, NUM_FEATURES, 
                    SEQ_LEN, OUT_LEN)
from scripts.utils import batch_extraction
from fastapi import HTTPException, status
from models.tcn import TCNPredictor
import litserve as ls 
import pandas as pd
import torch
import json
import time


class SimpleLogger(ls.Logger):
    def process(self, key, value):
        print(f"Received {key} with value {value}", flush=True)

class FileLogger(ls.Logger):
    def process(self, key, value):
        with open(PROJECT_ROOT+"/logs/api_logs/litserve_metrics.log", "a+") as f:
            f.write(f"{key}: {value:.1f}\n")
       
class ForecastAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.model = TCNPredictor(NUM_FEATURES, SEQ_LEN, OUT_LEN).to(device)
        self.model.load_state_dict(torch.load(MODEL_WEIGHTS, weights_only=True, map_location=device))
        self.model.eval()
    
    def decode_request(self, request, **kwargs):
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
        return batch
    
    def predict(self, x, **kwargs):
        t0 = time.time()
        prediction = self.model.predict(x.to(self.device))
        t1 = time.time()
        self.log("model_time", t1 - t0)
        return prediction
    
    def encode_response(self, output, **kwargs):
        """Convert model output tensor into a JSON-serializable dict."""
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