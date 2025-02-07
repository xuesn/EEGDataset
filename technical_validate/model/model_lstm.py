import torch.nn as nn

class ModelLSTM(nn.Module):
    def __init__(self, fc_in_dim,
                 electrode_num=122, class_num=20,
                 hidden_size=122, num_layers=1):
        super(ModelLSTM, self).__init__()
        # lstm
        self.model_lstm = nn.Sequential(
            nn.LSTM(electrode_num, hidden_size, num_layers, 
                    bias=True, batch_first=True, )
        )
        # fc
        self.model_fc = nn.Sequential(
            nn.Linear(fc_in_dim, class_num),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        # bs * time * electrode
        lstm_out,  (hn, cn) = self.model_lstm(x)
        bs = lstm_out.shape[0]
        fc_in = lstm_out.reshape([bs,-1])
        return self.model_fc(fc_in)
    




