import os, sys, json
# Rooth directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ******* GLOBAL PARAM *******     
DEVICE = "cpu"

# ******* SERVER PARAMS ******* 
# Target Model to use
RUN = "2025_05_04_11_11_35" 

# Self explanatory
MODEL_WEIGHTS = PROJECT_ROOT+f"/models/weights/tcn_{RUN}.pt"

with open(PROJECT_ROOT+f"/logs/train_logs/run_{RUN}/train_state.json", 'r') as file:
    train_state = json.load(file)

# NUmber of input features that the model will recieve
NUM_FEATURES = len(train_state["time_varying_unknown_reals"])

# Length of the input sequence
SEQ_LEN = train_state["model_params"]["max_encoder_length"]

# Length of the output sequence
OUT_LEN = train_state["model_params"]["max_prediction_length"]