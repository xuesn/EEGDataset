import numpy as np
import torch

# train / val
def train(train_or_val_phase, dataloader, 
    model, optimizer, criterion_ce, rank, ):

    if train_or_val_phase == 'train':
        model.train()
    elif train_or_val_phase == 'test':
        model.eval()
    else:
        assert False
        
    sample_num = len(dataloader.dataset)
    batch_num = len(dataloader)
    total_loss = 0
    total_right_num = 0
    for step,  (eeg, label_onehot) in enumerate(dataloader):
        with torch.set_grad_enabled(train_or_val_phase == 'train'):
            eeg = eeg.to(rank)
            label_onehot = label_onehot.to(rank)
    
            prob = model(eeg)
            loss = criterion_ce(prob,  label_onehot) 

            pred = np.argmax(prob.cpu().detach().numpy(),  axis=1)
            label = np.argmax(label_onehot.cpu().detach().numpy(),  axis=1)    
            right_num_batch = np.sum(pred==label)

            if train_or_val_phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            total_loss += loss.cpu().detach().numpy()
            total_right_num += right_num_batch

    total_loss /= batch_num
    accuracy = total_right_num / sample_num
    return total_loss, accuracy
    







