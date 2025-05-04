""" import os, sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import torch
from models.tcn import TCNPredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    save_model_path = proj_root + "/models/weights/tcn_best.pt"
    model = TCNPredictor(input_size=13, seq_len=8, output_size=4).to(DEVICE)
    model.load_state_dict(torch.load(save_model_path, weights_only=True))
    
    random_input = torch.rand((3,8,13))
    model.eval()
    output = model(random_input.to(DEVICE))
    print(output.shape)
    print("success")

if __name__ == "__main__":
    main()
 """