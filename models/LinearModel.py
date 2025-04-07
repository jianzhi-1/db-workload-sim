import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, N, T) -> None:
        super().__init__()
        self.N = N
        self.T = T # max delay
        self.input_dim = N*(2*T+1)
        self.hidden_dim = int(np.sqrt(N))*2*T
        self.seq = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU()
        )
        self.lambda_head = nn.Linear(self.hidden_dim, (T+2))
        self.p_header = nn.Linear(self.hidden_dim, (T+1))
        self.lambda_output_dim = (T+2)
        self.p_output_dim = (T+1)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        assert x.shape == (batch_size, self.N, self.N, 2*self.T + 1)
        xp = torch.reshape(x, (batch_size, self.N, -1))
        assert xp.shape == (batch_size, self.N, self.input_dim)
        xq = self.seq(xp)
        assert xq.shape == (batch_size, self.N, self.hidden_dim)

        lambda_pre_softmax = nn.ELU()(self.lambda_head(xq))
        assert lambda_pre_softmax.shape == (batch_size, self.N, self.lambda_output_dim)
        lamb = F.softmax(lambda_pre_softmax, dim=2)
        assert lamb.shape == (batch_size, self.N, self.lambda_output_dim)

        p = nn.ELU()(self.p_header(xq))
        assert p.shape == (batch_size, self.N, self.p_output_dim)
        return lamb, p