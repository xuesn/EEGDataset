import torch.nn as nn

class ModelTransformer(nn.Module):
    def __init__(self, fc_in_dim,
                 electrode_num=122, class_num=20,
                 hidden_size=122, num_layers=1, 
                 in_channel_num=128, trans_layers=1, trans_head=8, trans_fc_hid=256                 
                 ):
        super(ModelTransformer, self).__init__()
        # linear
        self.in_channel_num = in_channel_num
        self.model_linear = nn.Sequential(
            nn.Linear(electrode_num, in_channel_num),
        )
        # transformer
        trans_enc_layer = nn.TransformerEncoderLayer(d_model=in_channel_num, 
                                                     nhead=trans_head, dim_feedforward=trans_fc_hid,
                                                     batch_first=True, )
        self.trans_tile = nn.TransformerEncoder(trans_enc_layer, num_layers=trans_layers)
        # fc
        self.model_fc = nn.Sequential(
            nn.Linear(fc_in_dim, class_num),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        # bs * time * electrode
        [bs, time_num, electrode_num] = x.shape
        x_reshape = x.reshape([bs*time_num, electrode_num])
        linear_out = self.model_linear(x_reshape)
        linear_out_reshape = linear_out.reshape([bs, time_num, self.in_channel_num])
        trans_out = self.trans_tile(linear_out_reshape)
        fc_in = trans_out.reshape([bs,-1])
        return self.model_fc(fc_in)

