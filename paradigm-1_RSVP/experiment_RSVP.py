from __future__ import division

from psychopy import visual, core, event
from psychopy.hardware import keyboard

import time

import serial

import os
import pandas as pd
import numpy as np

from utils_RSVP import rand_sample_multiclass_consecutive
from utils_RSVP import imgshow_serialwrite, select_serialwrite
from utils_RSVP import save_filepath_per_experiment_consecutive, save_exp_info
from utils_RSVP import rand_special_img_seq_and_pos, save_special_img_seq_and_pos

import gc

#
##
###--SPECIAL--###
exp_date = ''
sub_name = ''  
expNO = 

###--SPECIAL--###
##
#
run_start = 0  
whether_resample = True  

# ----------------------------------------------------------------------------------------------------
exp_dir_path = 'C:\\Users\\Desktop\\eeg'
pic_path = exp_dir_path + '\\pic_10000_resized\\'  
special_img_dir = exp_dir_path+'\\pic_10000_resized\\special\\' 
sample_num_per_class = 100  
img_num_per_break = 20  
run_num = 2
special_img_num_per_run = 10  
sequence_num = 50 
class_strlist_yesno = ['yes', 'no', ]
class_strlist_english = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                         'bus', 'car', 'cat', 'chair', 'cow',
                         'diningtable', 'dog', 'flower', 'horse', 'motorbike',
                         'person', 'sheep', 'sofa', 'train', 'tvmonitor']
class_strlist_chinese = ['飞机', '自行车', '鸟', '船', '瓶子',
                         '巴士', '汽车', '猫', '椅子', '牛',
                         '餐桌', '狗', '花', '马',  '助力车',
                         '人', '羊', '沙发', '火车', '显示器']
class_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
              'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
              'diningtable': 10, 'dog': 11, 'flower': 12, 'horse': 13, 'motorbike': 14,
              'person': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19,
              '飞机': 0, '自行车': 1, '鸟': 2, '船': 3, '瓶子': 4,
              '巴士': 5, '汽车': 6, '猫': 7, '椅子': 8, '牛': 9,
              '餐桌': 10, '狗': 11, '花': 12, '马': 13, '助力车': 14,
              '人': 15, '羊': 16, '沙发': 17, '火车': 18, '显示器': 19,
              'special': 99}
class_num = len(class_strlist_english)
piclist_savepath = exp_dir_path+'\\paradigm_1_RSVP\\img_info_RSVP\\'
filepath_save_name = exp_date+sub_name + \
    str(expNO)+'_sample'+str(sample_num_per_class) + \
    'class'+str(class_num)+'_RSVP.csv'
special_img_savepath = exp_dir_path + \
    '\\paradigm_1_RSVP\\special_img_info_RSVP\\'
special_img_save_name = exp_date+sub_name + \
    str(expNO)+'_special_img_num'+str(special_img_num_per_run) + \
    '_RSVP.csv'


# ----------------------------------------------------------------------------------------------------
if whether_resample:
    print('Start img sampling!')
    img_list = rand_sample_multiclass_consecutive(
        pic_path, sample_num_per_class, img_num_per_break, class_strlist_english)
    print('Finished img sampling!')
    save_filepath_per_experiment_consecutive(
        piclist_savepath, filepath_save_name, img_list)
    print('Start special-img sampling!')
    [special_seqNO_list, insert_pos_list, special_img_path_list] = rand_special_img_seq_and_pos(
        run_num, special_img_num_per_run, img_num_per_break, sequence_num, special_img_dir)
    print('Finished special-img sampling!')
    save_special_img_seq_and_pos(
        special_img_savepath, special_img_save_name, special_seqNO_list, insert_pos_list, special_img_path_list)
else: 
    print('Start read img_list and special_seqNO_list, insert_pos_list, special_img_path_list !')
    full_path = os.path.join(piclist_savepath, filepath_save_name)
    data_ori = pd.read_table(full_path, sep=",", header=0)
    index_numpy = data_ori.index.to_numpy()
    index_range = np.ones([len(index_numpy), ], dtype=np.bool)
    index_range[::img_num_per_break+1] = 0
    img_list = index_numpy[index_range].reshape(-1, img_num_per_break).tolist()
    full_path = os.path.join(special_img_savepath, special_img_save_name)
    data_ori = pd.read_table(full_path, sep=",", index_col=False, names=[
                             'seq', 'pos', 'path'])
    value_numpy = data_ori.to_numpy()
    index_range = np.ones([len(value_numpy), ], dtype=np.bool)
    index_range[::special_img_num_per_run+1] = 0
    seq_pos_path = value_numpy[index_range]
    special_seqNO_list = []
    insert_pos_list = []
    special_img_path_list = []
    for runNO in range(run_num):
        special_seqNO_list.append(seq_pos_path[runNO*special_img_num_per_run:(
            runNO+1)*special_img_num_per_run, 0].astype(np.int).tolist())
        insert_pos_list.append(seq_pos_path[runNO*special_img_num_per_run:(
            runNO+1)*special_img_num_per_run, 1].astype(np.int).tolist())
        special_img_path_list.append(seq_pos_path[runNO*special_img_num_per_run:(
            runNO+1)*special_img_num_per_run, 2].astype(np.str).tolist())
    print('Finished read img_list and special_seqNO_list, insert_pos_list, special_img_path_list !')

img_ser_write_range = np.arange(start=1, stop=img_num_per_break+1)
img_ser_write_array = np.tile(img_ser_write_range.reshape(
    1, -1), reps=(run_num*sequence_num, 1))
img_ser_write_list = img_ser_write_array.tolist()
special_img_ser_write = 99  
for runNO in range(run_num):
    special_seqNO_inner_list = special_seqNO_list[runNO]
    insert_pos_inner_list = insert_pos_list[runNO]
    special_img_path_inner_list = special_img_path_list[runNO]
    for seqNO in range(sequence_num):
        is_contain_special_img = False
        if seqNO in special_seqNO_inner_list:
            index = special_seqNO_inner_list.index(
                seqNO)  
            is_contain_special_img = True
            insert_pos = insert_pos_inner_list[index]
            special_img_path = special_img_path_inner_list[index]
            img_list[runNO*sequence_num +
                     seqNO].insert(insert_pos, special_img_path)
            img_ser_write_list[runNO*sequence_num +
                               seqNO].insert(insert_pos, special_img_ser_write)

# end----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
fresh_rate = 60
singleobj_time_blank_before_after_sequence = 0.75
singleobj_frameN_blank_before_after_sequence = int(
    singleobj_time_blank_before_after_sequence * fresh_rate)
singleobj_time_imgshow = 0.1  # picture present time
singleobj_frameN_imgshow = int(singleobj_time_imgshow * fresh_rate)
singleobj_time_blank_afterimg = 0.1  # picture present  ISI time
singleobj_frameN_blank_afterimg = int(
    singleobj_time_blank_afterimg * fresh_rate)
singleobj_time_respond = 2
singleobj_frameN_respond = int(singleobj_time_respond * fresh_rate)
# end----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
ser = serial.Serial('COM4', 115200, timeout=1)
for i in range(1,21):
    ser.write([0])
win = visual.Window(size=(1920, 1080), fullscr=False, screen=0, allowGUI=False, allowStencil=False,
                    monitor='testMonitor', color=[-1, -1, -1], colorSpace='rgb', blendMode='avg', useFBO=True)
# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()
# myMouse = event.Mouse()
# end----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
intro = visual.TextStim(win, text=u'按空格开始', height=0.12,
                        pos=(-0.1, 0.1), bold=True, italic=False, color='white')
intro.draw()
win.flip()
k_1 = event.waitKeys(keyList=['space'])
print(k_1)

exp_info_savepath = exp_dir_path+'\\paradigm_1_RSVP\\exp_info_RSVP\\'
exp_info_filename = exp_date+sub_name+str(expNO)+'_exp_info_RSVP.csv'

save_list_row_start = ['experiment-start!', time.time(), time.asctime(
    time.localtime(time.time()))]  
save_exp_info(exp_info_savepath, exp_info_filename, save_list_row_start)

right_number = 0  
for runNO in range(run_start, run_num):
    save_list_row_start = ['run'+str(runNO+1)+'-start!', time.time(), time.asctime(
        time.localtime(time.time()))]  
    save_exp_info(exp_info_savepath, exp_info_filename,
                  save_list_row_start)

    special_seqNO_inner_list = special_seqNO_list[runNO]
    insert_pos_inner_list = insert_pos_list[runNO]
    for seqNO in range(sequence_num):
        if seqNO in special_seqNO_inner_list:
            index = special_seqNO_inner_list.index(
                seqNO)  
            is_contain_special_img = True
            insert_pos = insert_pos_inner_list[index]
        else:
            is_contain_special_img = False

        img_list_seq = img_list[runNO*sequence_num+seqNO]
        img_ser_write_list_seq = img_ser_write_list[runNO*sequence_num+seqNO]

        # pic_all_win = []
        # del pic_all_win
        pic_all_win = []
        gc.collect()
        print('Start ImageStim preload!')
        for i in range(len(img_list_seq)):
            pic_all_win.append(visual.ImageStim(win, image=img_list_seq[i], mask=None, units='', pos=(0.0, 0.0), size=None, ori=0.0, color=(1.0, 1.0, 1.0),
                                                colorSpace='rgb', contrast=1.0, opacity=1.0, depth=0, interpolate=False, flipHoriz=False, flipVert=False, texRes=128, name=None, autoLog=None, maskParams=None))
        print('Finished ImageStim preload!')

        # --------------------------------------------------
        for frameN in range(singleobj_frameN_blank_before_after_sequence):
            white_cross = visual.TextStim(win, text=u'+', height=0.20,
                                          pos=(0, 0), bold=True, italic=False, color='white')
            white_cross.draw()
            win.flip()

        # --------------------------------------------------
        for imgNO, img_obj in enumerate(pic_all_win):
            pic_win = img_obj
            imgNO_show_write = img_ser_write_list_seq[imgNO]
            print('\n####'+'runNO-'+str(runNO+1) +
                  ' sequenceNO-'+str(seqNO+1)+'####')
            print('####imageNO-'+str(imgNO_show_write)+'####')
            imgshow_time = time.time()
            imgshow_localtime_str = time.asctime(time.localtime(time.time()))
            imgshow_serialwrite(
                pic_win, imgNO_show_write, win, singleobj_frameN_imgshow, singleobj_frameN_blank_afterimg, ser)
            img_path = img_list_seq[imgNO]
            path_split_str = '\\'  
            class_str = img_path.split(
                path_split_str)[-2]  
            img_class = class_dict[class_str]
            print('        ', class_str)
            save_list_row_imgshow = [imgNO_show_write, img_path,
                                     class_str, img_class, imgshow_time, imgshow_localtime_str, ]
            save_exp_info(exp_info_savepath, exp_info_filename,
                          save_list_row_imgshow)

        # --------------------------------------------------
        for frameN in range(singleobj_frameN_blank_before_after_sequence):
            win.flip()

        # --------------------------------------------------
        print('\nBREAK-TIME')
        is_contain_special_img = (len(img_list_seq) == (
            img_num_per_break+1))  
        choice_class, flag_correct = select_serialwrite(
            win, defaultKeyboard,  is_contain_special_img,  singleobj_frameN_respond, ser)
        choice_num = 2
        if choice_class < choice_num:
            choice_class_str = class_strlist_yesno[choice_class]
        elif choice_class == choice_num:
            choice_class_str = 'NONE'
        else:
            assert 'no such choice!!'
        print('SELECT: ', choice_class_str)

        select_time = time.time()
        select_localtime_str = time.asctime(time.localtime(time.time()))
        right_number += int(flag_correct)  

        for frameN in range(singleobj_frameN_blank_before_after_sequence):
            white_cross = visual.TextStim(win, text=u'+', height=0.20,
                                          pos=(0, 0), bold=True, italic=False, color='white')
            white_cross.draw()
            win.flip()

        save_list_row_break = ['run'+str(runNO+1)+'sequence'+str(seqNO+1)+'-break-time!',
                               time.time(), time.asctime(time.localtime(time.time())),
                               choice_class,  str(flag_correct),   right_number] 
        save_exp_info(exp_info_savepath, exp_info_filename,
                      save_list_row_break)

        # --------------------------------------------------
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            print('\nQUIT-GAME')
            ser.close()
            win.close()
            core.quit()

    intro = visual.TextStim(win, text=u'请休息', height=0.12,
                            pos=(-0.1, 0.1), bold=True, italic=False, color='white')
    intro.draw()
    win.flip()
    run_num_per_long_sleep = 2
    if ((runNO+1) % run_num_per_long_sleep == 0) & (runNO != 9):
        time.sleep(2)  
    else:
        time.sleep(5)  
    # winsound.Beep(1000, 500)
    intro = visual.TextStim(win, text=u'按空格开始', height=0.12,
                            pos=(-0.1, 0.1), bold=True, italic=False, color='white')
    intro.draw()
    win.flip()
    k_1 = event.waitKeys(keyList=['space'])  
    # end----------------------------------------------------------------------------------------------------
    
print(f"correct rate: {right_number/run_num/sequence_num}")
print('\nQUIT-GAME')
ser.close()
win.close()
core.quit()
