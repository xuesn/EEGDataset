import torch.nn as nn

class ModelLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ModelLinear, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim,out_dim)
        )
    def forward(self, x):
        # bs * time * electrode
        x_flat = x.reshape(x.shape[0], -1)
        return self.model(x_flat)
