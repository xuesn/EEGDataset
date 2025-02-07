import cv2
import pandas as pd
import os
import random

import pyttsx3
from psychopy import visual

import numpy as np


def change_img_size_thisdir(ori_path, save_path):
    fileList = os.listdir(ori_path)
    for fileName in fileList:  
        src = os.path.join(ori_path, fileName) 
        dst = os.path.join(save_path, fileName.split('.')
                           [0]+'_resized' + '.jpg')
        image = cv2.imread(src)
        height_resize = 640
        height, width = image.shape[0], image.shape[1]
        scale = height_resize / height
        width_resize = int(width * scale)
        # resize
        image_resize = cv2.resize(image, (width_resize, height_resize))
        try:
            cv2.imwrite(dst, image_resize)
            print('converting %s to %s ...' % (src, dst))
        except:
            continue
    return


def change_img_size_multidir(dir_path, save_path):
    dirname_list = os.listdir(dir_path)
    for dirname in dirname_list:
        thisclass_ori_path = os.path.join(dir_path, dirname)
        thisclass_save_path = os.path.join(save_path, dirname)
        if not os.path.exists(thisclass_save_path):
            os.makedirs(thisclass_save_path)
        change_img_size_thisdir(thisclass_ori_path, thisclass_save_path)
    return


def rand_sample_filename_consecutive(dir_path, sample_num, img_num_per_break,):
    filelist = os.listdir(dir_path)
    filelist_sampled = random.sample(filelist, sample_num)
    random.shuffle(filelist_sampled)
    filelist_sampled_fullpath = [os.path.join(dir_path, file)
                                 for file in filelist_sampled]

    sequence_num = sample_num/img_num_per_break  
    filelist_sampled_arr = np.array(filelist_sampled_fullpath)
    filelist_sampled_consecutive = filelist_sampled_arr.reshape(
        [-1, img_num_per_break]).tolist()
    return filelist_sampled_consecutive


def rand_sample_multiclass_consecutive(dir_path, sample_num_per_class, img_num_per_break, class_list):
    filepath_list_multiclass = []
    for class_name in class_list:
        class_dir = os.path.join(dir_path, class_name)
        filelist_sampled_this_class = rand_sample_filename_consecutive(
            class_dir, sample_num_per_class, img_num_per_break,)
        filepath_list_multiclass += filelist_sampled_this_class 
    random.shuffle(filepath_list_multiclass) 
    return filepath_list_multiclass


def save_filepath_per_experiment_consecutive(save_path, save_name, filepath_list_multiclass):
    full_path = os.path.join(save_path, save_name)
    assert not os.path.exists(full_path), '文件已存在，避免覆盖，代码终止！！'
    columns = ['FilePath']
    df = pd.DataFrame(data=columns)
    df.to_csv(full_path, mode='a',
              header=False, index=False) 
    for seqNO, filepath_list_single_seq in enumerate(filepath_list_multiclass):
        df = pd.DataFrame(data=[['Sequence:', str(seqNO+1)]])
        df.to_csv(full_path, mode='a',
                  header=False, index=False)  
        df = pd.DataFrame(data=filepath_list_single_seq)
        df.to_csv(full_path, mode='a',
                  header=False, index=False) 
    return


def imgshow_serialwrite(pic_win, imgNO, win, singleobj_frameN_imgshow, singleobj_frameN_blank_afterimg, ser):
    ser.write([0])
    ser.write([imgNO])  
    for i in range(1,20):
        ser.write([0])

    for frameN in range(singleobj_frameN_imgshow):
        win.flip()
        pic_win.draw()
    for frameN in range(singleobj_frameN_blank_afterimg):
        win.flip()
    return


def select_serialwrite(win, defaultKeyboard,  is_contain_special_img,  singleobj_frameN_respond, ser):
    is_clicked = False
    choice_class = 2 
    defaultKeyboard.clearEvents()  
    for frameN in range(singleobj_frameN_respond):
        if not is_clicked:
            # if defaultKeyboard.getKeys(keyList=["y"], waitRelease=True, clear=True):
            #     choice_class = 0  # yes
            #     is_clicked = True
            # if defaultKeyboard.getKeys(keyList=["n"], waitRelease=True, clear=True):
            #     choice_class = 1  # no
            #     is_clicked = True
            if defaultKeyboard.getKeys(keyList=["left"], waitRelease=False, clear=True):
                choice_class = 0  # yes
                flag_correct = is_correct_feedback(
                    choice_class, is_contain_special_img, ser)  
                is_clicked = True
            if defaultKeyboard.getKeys(keyList=["right"], waitRelease=False, clear=True):
                choice_class = 1  # no
                flag_correct = is_correct_feedback(
                    choice_class, is_contain_special_img, ser)  
                is_clicked = True
        intro = visual.TextStim(win, text=u'先眨眼，再反馈', height=0.12,
                                pos=(0, 0.5), bold=True, italic=False, color='white')
        intro.draw()
        win.flip()
    defaultKeyboard.clearEvents() 
    if not is_clicked:
        flag_correct = is_correct_feedback(
            choice_class, is_contain_special_img, ser)  
    return choice_class, flag_correct


def is_correct_feedback(choice_class, is_contain_special_img, ser):
    if choice_class == int(not is_contain_special_img):
        ser.write([0])
        ser.write([252])  # means correct
        for i in range(1,20):
            ser.write([0])
        flag_correct = True
        # pyttsx3.speak('right')
    else:
        ser.write([0])
        ser.write([254])  # means wrong      
        for i in range(1,20):
            ser.write([0])
        flag_correct = False
        # pyttsx3.speak('wrong')
    return flag_correct


def save_exp_info(exp_info_savepath, exp_info_filename, save_list_row):
    full_path = os.path.join(exp_info_savepath, exp_info_filename)
    if not os.path.exists(full_path):
        columns = ['imgNO', 'img_path',
                   'class_str', 'img_class',   'imgshow_time',   'imgshow_localtime_str', ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(full_path, index=False)
        select_list = ['', 'time',
                       'localtime_str', 'choice',   'flag_correct',   'right_number', ]
        df = pd.DataFrame(data=[select_list])
        df.to_csv(full_path, mode='a',
                  header=False, index=False)
    df = pd.DataFrame(data=[save_list_row])  
    df.to_csv(full_path, mode='a',
              header=False, index=False) 
    return


def rand_special_img_seq_and_pos(run_num, special_img_num, img_num_per_break, sequence_num, special_img_dir):
    special_seqNO_list = []
    insert_pos_list = []
    special_img_path_list = []
    for runNO in range(run_num):
        sequence_arange = np.arange(sequence_num)
        # Chooses k unique random elements from a population sequence or set.
        special_seqNO_list.append(
            random.sample(population=sequence_arange.tolist(), k=special_img_num))

        position_arange = np.arange(img_num_per_break+1)
        # Return a k sized list of population elements chosen with replacement.
        insert_pos_list.append(
            random.choices(population=position_arange.tolist(), k=special_img_num))

        special_img_filelist = os.listdir(special_img_dir)
        filelist_special_img_fullpath = [os.path.join(special_img_dir, file)
                                         for file in special_img_filelist]
        special_img_path_list.append(
            random.choices(population=filelist_special_img_fullpath, k=special_img_num))

    return special_seqNO_list, insert_pos_list, special_img_path_list


def save_special_img_seq_and_pos(save_path, save_name, special_seqNO_list, insert_pos_list, special_img_path_list):
    full_path = os.path.join(save_path, save_name)
    assert not os.path.exists(full_path), 'The file already exists, avoid overwriting, and the code terminates！！'
    for runNO, special_seqNO_inner_list in enumerate(special_seqNO_list):
        insert_pos_inner_list = insert_pos_list[runNO]
        special_img_path_inner_list = special_img_path_list[runNO]
        df = pd.DataFrame(['runNO'+str(runNO)])
        df.to_csv(full_path, mode='a',
                  header=False, index=False)  
        df = pd.DataFrame({'special_sequenceNO': special_seqNO_inner_list,
                           'insert_position': insert_pos_inner_list,
                           'special_img_path': special_img_path_inner_list})
        df.to_csv(full_path, mode='a',
                  header=False, index=False)  
    return
