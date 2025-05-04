import os, sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from starlette.datastructures import FormData, UploadFile
from fastapi import HTTPException, status
from scripts.utils import batch_extraction
from models.tcn import TCNPredictor
import litserve as ls 
import pandas as pd
import numpy as np
import torch
import json

RUN = "2025_05_04_11_11_35" #"2025_05_03_22_24_58"
MODEL_WEIGHTS = proj_root+f"/models/weights/tcn_{RUN}.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ForecastAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.model = TCNPredictor(input_size=13, seq_len=8, output_size=4).to(device)
        self.model.load_state_dict(torch.load(MODEL_WEIGHTS, weights_only=True))
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
        print("decode_request_step")
        print(batch)
        return batch
    
    #def batch(self, inputs):
    #    return inputs
    
    def predict(self, x, **kwargs):
        predictions = self.model.predict(x.to(DEVICE))
        print("predict_step")
        print(predictions.shape)
        return predictions
    
    #def unbatch(self, output):
    #    return output
    
    def encode_response(self, output, **kwargs):
        print("encode_response_step")
        if self.device != "cpu":
            output = output.cpu().numpy()
        else:
            output = output.numpy()
        
        product_output = {}
        for i, product_id in enumerate(self.product_ids):
            weeks_output = {}
            for week in range(output.shape[1]):
                weeks_output["week_"+str(week)] = str(output[i,week]) 
            product_output[str(product_id)] = weeks_output

        print("almost_done")
        print(product_output)
        return product_output

if __name__ == "__main__":
    api = ForecastAPI()
    server = ls.LitServer(api, accelerator=DEVICE)
    server.run(port=8000)