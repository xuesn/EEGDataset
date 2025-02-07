import os
import mne
import pandas as pd
import numpy as np
import json


# parameter
paradigm_str = 'task-lowSpeed'

# downsample_freq = 250
downsample_freq = 1000
freq_low = 0.1
freq_high = 100
if freq_high>50 and freq_low<50:
    freq_notch = 50
else:
    freq_notch= None

# current epoch
epoch_st = 0
epoch_end = 0.5  
baseline_st = 0
baseline_end = 0

# another epoch
# epoch_st = -0.1  
# epoch_end = 1  
# baseline_st = -0.1
# baseline_end = 0

amplify_rate = 1_000_000  # V-->uV
overwrite_flag = 0

preprocessed_path = './derivatives/preprocessed_data/'
raw_dataset_path = './'

sub_list = ['sub-01']
sub_list.sort()
for sub_dir in sub_list:
    sub_path = os.path.join(raw_dataset_path,sub_dir)
    ses_list = os.listdir(sub_path)
    ses_list.sort()
    for ses_dir in ses_list:
        ses_path = os.path.join(sub_path,ses_dir)
        file_list = os.listdir(ses_path)
        cdt_list = [fn for fn in file_list if fn.endswith('.cdt')]
        cdt_list = [fn for fn in cdt_list if paradigm_str in fn]
        cdt_list.sort()
        for cdt_fname in cdt_list:
            file_prefix = cdt_fname.split('_eeg.cdt')[0]
            exp_info_fname = file_prefix + '_record.csv'
            cdt_path = os.path.join(ses_path,cdt_fname)
            exp_info_path = os.path.join(ses_path,exp_info_fname)
            
            # load data
            raw_data = mne.io.read_raw_curry(cdt_path, verbose='warning')
            raw_data.drop_channels(ch_names=['10', '11', '84', '85',  '110', '111',
                                                    'VEO', 'HEO', 'EKG', 'EMG', 'Trigger', ]) 
            raw_data.load_data()
            # raw_data.set_eeg_reference(ref_channels='average')
            raw_data.filter(l_freq=freq_low, h_freq=freq_high,
                            fir_design='firwin', verbose='warning')
            if freq_notch is not None:
                raw_data.notch_filter(
                    np.arange(freq_notch, freq_high, freq_notch), verbose='warning') 
            raw_data.resample(sfreq=downsample_freq, verbose='warning')
            
            # load events         
            events_fname = file_prefix+'_events.tsv' 
            events_info = pd.read_table(os.path.join(ses_path,events_fname), sep='\t', header=0)  
            event_moment_list = events_info['onset'].values     
            event_key_list = events_info['eventnumber'].values    
            # 
            events_manual = np.zeros([len(event_key_list),3],dtype=np.int32)
            events_manual[:,0]=(event_moment_list/(1000/downsample_freq)).astype('int')
            events_manual[:,2]=event_key_list
            event_id_epoch_low_speed= [i for i in range(1,100+1)]
            
            # epoch
            if baseline_st==baseline_end:
                baseline=None
            else:
                baseline=(baseline_st, baseline_end)
            duration_sec = epoch_end-epoch_st
            timepoint_num = int(duration_sec*downsample_freq)
            epochs = mne.Epochs(raw_data, events_manual, event_id=event_id_epoch_low_speed, tmin=epoch_st, tmax=epoch_end,
                                baseline=baseline, picks=None, preload=False,
                                reject=None, flat=None, proj=True, decim=1,
                                reject_tmin=None, reject_tmax=None, detrend=None,
                                on_missing='raise', reject_by_annotation=True, metadata=None,
                                event_repeated='error', verbose=None)
            epochs.load_data()
            numpy_data = epochs.get_data()
            # del raw_data
            # del epochs
            sample_time_electrode = numpy_data.transpose(0, 2, 1) 
            sample_time_electrode = sample_time_electrode[:, :-1, :]  # deleted last timepoint (126-->125)
            sample_time_electrode *= amplify_rate

            # load low-speed-label
            exp_info = pd.read_csv(exp_info_path, sep=',',
                                header=0)  
            img_class = exp_info['img_class'].values
            # remove NaN
            img_class = img_class.astype('float')
            not_nan_index = ~np.isnan(img_class)
            img_class = img_class[not_nan_index]
            img_class = img_class.astype('int')

            # save_path
            save_dir=os.path.join(preprocessed_path,sub_dir,ses_dir,)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(save_dir, ' created!')
            #
            data_save_fname=file_prefix+'_'+str(downsample_freq)+'Hz.npy'
            data_save_path=os.path.join(save_dir,data_save_fname)
            if (not os.path.exists(data_save_path)) or (overwrite_flag == 1): 
                np.save(data_save_path,sample_time_electrode.astype(np.float32))
            #
            label_save_fname=file_prefix+'_'+str(downsample_freq)+'Hz.json'
            label_save_path=os.path.join(save_dir,label_save_fname)
            if (not os.path.exists(label_save_path)) or (overwrite_flag == 1): 
                with open(label_save_path, "w") as f:
                    json.dump(img_class.tolist(), f, indent=2)
            print('{} saved!'.format(file_prefix))




