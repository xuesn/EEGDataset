
import os
import pandas as pd

summarize_save_dir = ""
if not os.path.exists(summarize_save_dir):
    os.makedirs(summarize_save_dir)

loss_save_path = ""

file_list = os.listdir(loss_save_path)
csv_list = [fn for fn in file_list if fn.endswith(".csv")]

summarize_save_path = os.path.join(summarize_save_dir,
                'summarize_fnum'+str(len(csv_list))+'.csv')
for filename in csv_list:
    if not os.path.exists(summarize_save_path):  
        loss_acc_str_list = ['test Accuracy'] 
        para_name_list =  ['subject','session','model','paradigm',]
        col_str_list = loss_acc_str_list + para_name_list
        df = pd.DataFrame(columns=col_str_list)
        df.to_csv(summarize_save_path, index=False)

    # parameter
    para_list = filename.split('_.csv')[0].split('_')
    sub_str = para_list[0]
    ses_str = para_list[1]
    mod_str = para_list[2]
    par_str = para_list[3]

    # read
    csv_path = os.path.join(loss_save_path, filename)
    data_ori = pd.read_table(csv_path, sep=",", header=0)
    test_acc = data_ori['test Accuracy'].to_numpy()
    # max-accuracy
    if len(test_acc)<50:
        continue
    max_test_accuracy = max(test_acc)

    # 
    list_basic = [max_test_accuracy] + para_list
    data = pd.DataFrame(
        [list_basic])  
    data.to_csv(summarize_save_path, mode='a',
                header=False, index=False)  

print(summarize_save_path,' saved!')