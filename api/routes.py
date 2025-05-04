import os, sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from starlette.datastructures import FormData, UploadFile
from scripts.utils import batch_preprocessing, batch_extraction
from models.tcn import TCNPredictor
import litserve as ls 
import pandas as pd
import numpy as np
import torch
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN = "2025_05_03_22_24_58"
MODEL_WEIGHTS = proj_root+f"/models/weights/tcn_{RUN}.pt"

class ForecastAPI(ls.LitAPI):
    def setup(self, device):
        self.model = TCNPredictor(input_size=13, seq_len=8, output_size=4).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_WEIGHTS, weights_only=True))
        #print("setup: ", self.model.device.type)

    def decode_request(self, request, **kwargs):
        # Product IDs extraction
        metadata_str = request.get("metadata")
        metadata = json.loads(metadata_str)
        product_ids = metadata["product_id"]
        if len(product_ids) == 0:
            raise Exception("No products id were given. Provide 1 at least.") 
        print(product_ids)
        
        # Dataframe extraction
        df_upload = request.get("file")
        df = pd.read_csv(df_upload.file)
        batch = batch_extraction(df, product_ids)
        #df = df[df["Product_ID"].isin(product_ids)]
        # total_amount, Age, Male, Female, Quantity, price_per_unit, year, month, week, window_mean_4, window_mean5, window_mean_6, window_mean7, 
        print("decode_request_step")
        print(batch.shape)
        #print(f"decoded_request: ",type(image))
        return "image"
    
    #def batch(self, inputs):
    #    print("batch_step")
    #    #print("batch: ", type(inputs))
    #    return list(inputs)
    
    def predict(self, x, **kwargs):
        #clean_results = []
        print("predict_step")
        #print("predict: ", type(results))
        return x #[0]
    
    #def unbatch(self, output):
    #    print("unbatch_step")
    #    return output
    
    def encode_response(self, output, **kwargs):
        print("encode_response_step")
        print("========================================")
        return {"demo": "yes"}

if __name__ == "__main__":
    api = ForecastAPI()
    server = ls.LitServer(api, accelerator="auto", max_batch_size=8, batch_timeout=0.05)
    server.run(port=8000)