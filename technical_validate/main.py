import os
cudaNO_list = [  ]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cudaNO_list))  

import pandas as pd
import torch
from torch import optim
from torch import nn
from dataset_iva23 import Dataset_iva23
from model.model_linear import ModelLinear
from model.model_fc import ModelFC
from model.model_conv import ModelConv
from model.model_lstm import ModelLSTM
from model.model_transformer import ModelTransformer
from train import train


paradigm_str = 'RSVP'
paradigm_dir = 'paradigm-1_RSVP'
# paradigm_str = 'low-speed'
# paradigm_dir = 'paradigm-2_low-speed'

subNO = 
sesNO = 
subNO_str = 'sub-'+str(subNO).zfill(2)
sesNO_str = 'ses-'+str(sesNO).zfill(2)
train_runNO_list = [1,2,3]
if paradigm_str=='RSVP' and subNO==26 and sesNO==5:
    train_runNO_list = [1,2,]
test_runNO_list = [4]
    
save_path_csv = ''
mod_str = 'linear'
# mod_str = 'fc'
# mod_str = 'conv'
# mod_str = 'lstm'
# mod_str = 'transformer'
csv_fname = '_'.join([subNO_str,sesNO_str,mod_str,paradigm_str,'.csv'])

# parameter
rank = 0
learn_rate = 5e-5
start_epoch = 0
end_epoch = 100
criterion_ce = nn.CrossEntropyLoss()  

# dataloader
train_eeg_path_list = []
train_label_path_list = []
trainset = Dataset_iva23(train_eeg_path_list, train_label_path_list,)
test_eeg_path_list = []
test_label_path_list = []
testset = Dataset_iva23(test_eeg_path_list, test_label_path_list,)

train_batch_size = len(trainset)
test_batch_size = len(testset)
trainloader = torch.utils.data.DataLoader(trainset, train_batch_size,
                                        shuffle=True)
testloader = torch.utils.data.DataLoader(testset, test_batch_size,
                                        shuffle=False)
dataloaders = {'train':    trainloader,
                'test':    testloader, }   

# model
# # linear
# in_dim = 500*122
# model = ModelLinear(in_dim, out_dim=20)
# # fc
# in_dim = 500*122
# model = ModelFC(in_dim, hid_dim=256, out_dim=20,)
# # conv
fc_in_dim = 60*512
model = ModelConv(fc_in_dim,
                electrode_num=122, class_num=20,
                ch1=128, ch2=256, ch3=512, 
                kernal1=3, kernal2=3, kernal3=3, 
                )
# # LSTM
# fc_in_dim = 500*122
# model = ModelLSTM(fc_in_dim,
#                 electrode_num=122, class_num=20,
#                 hidden_size=122, num_layers=1)
# # Transformer
# fc_in_dim = 500*128
# model = ModelTransformer(fc_in_dim,
#                 electrode_num=122, class_num=20,
#                 hidden_size=122, num_layers=1, 
#                 in_channel_num=128, trans_layers=1, trans_head=8, trans_fc_hid=256                 
#                 )
model = model.to(rank)
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

# save
if not os.path.exists(save_path_csv):
    os.makedirs(save_path_csv)
    print(save_path_csv, ' created!')
full_path = os.path.join(save_path_csv, csv_fname)
columns = ['epoch', 'test Loss', 'test Accuracy', 
            'train Loss', 'train Accuracy',]
df = pd.DataFrame(columns=columns)
df.to_csv(full_path, index=False)

# train
max_acc = 0 
for epoch in range(start_epoch, end_epoch):
    save_list = [epoch]
    print("epoch:{}".format(epoch)) 
    for train_or_val_phase in ['test', 'train']:
        dataloader = dataloaders[train_or_val_phase]
        total_loss, accuracy = train(train_or_val_phase, dataloader, 
                                        model, optimizer, criterion_ce, rank,)
        print('    {}: \t total_loss:{:.6f} \t accuracy:{:.2f}%'.format(train_or_val_phase, total_loss, accuracy*100))
        save_list+=[total_loss, accuracy]

        if train_or_val_phase=='test' and accuracy>max_acc:
            max_acc=accuracy

    # save
    df = pd.DataFrame(data=[save_list])  
    df.to_csv(full_path, mode='a',
            header=False, index=False) 



