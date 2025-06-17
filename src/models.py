
import torch
import torch.nn as nn

class MLP(nn.Module):
    # A simple Multi-Layer Perceptron for our score/flow networks
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=2):
        super().__init__()
        # Input dimension is 3 (2 for state, 1 for time)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, t):
        # Concatenate time t as a feature
        t_reshaped = t.view(-1, 1).expand(-1, x.shape[0]).view(-1, 1) if t.numel() == 1 else t.view(-1,1)
        xt = torch.cat([x, t_reshaped], dim=1)
        return self.network(xt)
