
from pytorch_tcn import TCN
from torch import nn
import torch

class TCNPredictor(nn.Module):
    """
    Temporal Convolutional Network for sequence-to-future-value regression
    using pytorch-tcn library.
    Inputs:
      - x: Tensor of shape [batch, seq_len, features] (N, L, C)
    Outputs:
      - out: Tensor of shape [batch, output_size] predicting future values
    """
    def __init__(self,
                 input_size: int = 13,
                 seq_len: int = 12,
                 output_size: int = 8,
                 num_channels: list = [25, 25, 25, 25],
                 kernel_size: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.seq_len = seq_len
        # Instantiate TCN for input_shape='NLC'
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            input_shape='NLC'
        )
        
        self.flat = nn.Flatten()
        self.linear = nn.Linear(num_channels[-1]*self.seq_len, output_size)
        self.activ = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.flat(x)
        x = self.linear(x)
        return self.activ(x)
    
    def predict(self, x):
      x = self.forward(x).detach()
      print(x)
      return x