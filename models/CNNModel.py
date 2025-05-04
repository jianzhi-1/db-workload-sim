import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, N, T) -> None:
        super().__init__()
        self.N = N
        self.T = T  # max delay
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(2*T + 1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions
        conv_output_size = 128 * N * N
        
        # Fully connected layers for final predictions
        self.fc_lambda = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ELU(),
            nn.Linear(512, N * (T + 2))
        )
        
        self.fc_p = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ELU(),
            nn.Linear(512, N * (T + 1))
        )
        
        self.lambda_output_dim = (T + 2)
        self.p_output_dim = (T + 1)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        assert x.shape == (batch_size, self.N, self.N, 2*self.T + 1)
        
        # Reshape input for conv layers: [batch, channels, height, width]
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Apply convolutional layers
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        
        # Flatten for fully connected layers
        x = x.reshape(batch_size, -1)
        
        # Get lambda predictions
        lambda_pre_softmax = F.elu(self.fc_lambda(x))
        lambda_pre_softmax = lambda_pre_softmax.reshape(batch_size, self.N, self.lambda_output_dim)
        
        # Handle NaN values in lambda predictions
        lambda_pre_softmax = torch.nan_to_num(lambda_pre_softmax, nan=0.0)
        lamb = F.softmax(lambda_pre_softmax, dim=2)
        
        # Get p predictions
        p = F.elu(self.fc_p(x))
        p = p.reshape(batch_size, self.N, self.p_output_dim)
        
        # Handle NaN values in p predictions
        p = torch.nan_to_num(p, nan=0.0)
        
        return lamb, p
