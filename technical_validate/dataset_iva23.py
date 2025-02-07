
import numpy as np
import json
import torch


class Dataset_iva23(torch.utils.data.Dataset):
    def __init__(self,
                eeg_path_list, label_path_list,
                timepoint_num=500, electrode_num=122, class_num=20):
        eeg_dataset = np.zeros([0,timepoint_num,electrode_num])
        label_dataset = []

        for eeg_path in eeg_path_list:
            eeg_load = np.load(eeg_path)
            eeg_dataset = np.concatenate([eeg_dataset,eeg_load],axis=0) 
            
        for label_path in label_path_list:
            with open(label_path, "r") as f:
                label_load = json.load(f)
            label_dataset+=label_load
        sample_num = len(label_dataset)
        label_onehot_dataset = np.zeros([sample_num,class_num])
        for sampleNO, label in enumerate(label_dataset):
            label_onehot_dataset[sampleNO, label] = 1
            
        # clamp
        clamp_thres = 500  # 500 uV
        sample_num, time_num, electrode_num = eeg_dataset.shape
        sample_timeElectrode = eeg_dataset.reshape(
            [sample_num,-1])  
        sample_timeElectrode = (sample_timeElectrode - np.mean(
            sample_timeElectrode, axis=0))
        eeg_dataset = sample_timeElectrode.reshape(
            [sample_num, time_num, electrode_num])
        eeg_dataset[eeg_dataset >  clamp_thres] =  clamp_thres
        eeg_dataset[eeg_dataset < -clamp_thres] = -clamp_thres

        # normalize   per sample
        sample_num, time_num, electrode_num = eeg_dataset.shape
        sample_timeElectrode = eeg_dataset.reshape(
            [sample_num,-1])  
        sample_timeElectrode = (sample_timeElectrode - np.mean(
            sample_timeElectrode, axis=0)) / np.std(sample_timeElectrode, axis=0)
        eeg_dataset = sample_timeElectrode.reshape(
            [sample_num, time_num, electrode_num])
        
        # # normalize   per electrode
        # sample_num, time_num, electrode_num = eeg_dataset.shape
        # sample_time_electrode = eeg_dataset
        # sample_electrode_time = sample_time_electrode.transpose(0,2,1)
        # sampleElectrode_time = eeg_dataset.reshape(
        #     [sample_num*electrode_num,-1])  
        # sampleElectrode_time = (sampleElectrode_time - np.mean(
        #     sampleElectrode_time, axis=0)) / np.std(sampleElectrode_time, axis=0)
        # sample_electrode_time = sampleElectrode_time.reshape(
        #     [sample_num, electrode_num, time_num])
        # sample_time_electrode = sample_electrode_time.transpose(0,2,1)
        # eeg_dataset = sample_time_electrode 

        self.eeg_dataset = eeg_dataset
        self.label_onehot_dataset = label_onehot_dataset

    def __len__(self):
        return self.eeg_dataset.shape[0]

    def __getitem__(self, idx):  
        label_onehot=self.label_onehot_dataset[idx,:]
        eeg=self.eeg_dataset[idx,:,:]
        return torch.tensor(eeg).float(), torch.tensor(label_onehot).float()
        