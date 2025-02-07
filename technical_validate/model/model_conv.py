from torch import nn

class ModelConv(nn.Module):
    def __init__(self, fc_in_dim,
                 electrode_num=122, class_num=20,
                 ch1=128, ch2=256, ch3=512, 
                 kernal1=3, kernal2=3, kernal3=3, 
                 ):  
        super(ModelConv, self).__init__()       
        # conv
        self.model_conv = nn.Sequential(
            # layer1
            nn.Conv1d(in_channels=electrode_num, 
                      out_channels=ch1,
                      kernel_size=kernal1),
            nn.BatchNorm1d(num_features=ch1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            # layer2
            nn.Conv1d(in_channels=ch1, 
                      out_channels=ch2,
                      kernel_size=kernal2),
            nn.BatchNorm1d(num_features=ch2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            # layer3
            nn.Conv1d(in_channels=ch2, 
                      out_channels=ch3,
                      kernel_size=kernal3),
            nn.BatchNorm1d(num_features=ch3),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        # fc
        self.model_fc = nn.Sequential(
            nn.Linear(fc_in_dim, class_num),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        # bs * time * electrode
        x_transpose = x.transpose(dim0=1, dim1=2)
        # bs * electrode * time
        conv_out = self.model_conv(x_transpose)
        bs = conv_out.shape[0]
        fc_in = conv_out.reshape([bs,-1])
        return self.model_fc(fc_in)






