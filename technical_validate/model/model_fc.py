from torch import nn

class ModelFC(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim,):
        super(ModelFC, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(num_features=hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
            nn.Softmax(dim=-1),
        )  
    def forward(self, x):
        # bs * time * electrode
        x_flat = x.reshape(x.shape[0], -1)
        return self.model(x_flat)







