import cv2
import pandas as pd
import os
import random

import pyttsx3
from psychopy import visual


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


def rand_sample_filename(dir_path, sample_num):
    filelist = os.listdir(dir_path)
    filelist_sampled = random.sample(filelist, sample_num)
    return filelist_sampled


def rand_sample_multiclass(dir_path, sample_num_per_class, class_list):
    filepath_list_multiclass = []
    class_name = class_list[0]
    for class_name in class_list:
        class_dir = os.path.join(dir_path, class_name)
        filelist_sampled_this_class = rand_sample_filename(
            class_dir, sample_num_per_class)
        filepath_list = [os.path.join(class_dir, file)
                         for file in filelist_sampled_this_class]
        filepath_list_multiclass += filepath_list
    random.shuffle(filepath_list_multiclass)  
    return filepath_list_multiclass


def save_filepath_per_experiment(save_path, save_name, filepath_list_multiclass):
    full_path = os.path.join(save_path, save_name)
    assert not os.path.exists(full_path), 'The file already exists, avoid overwriting, and the code terminates！！'
    columns = ['FilePath']
    df = pd.DataFrame(data=columns)
    df.to_csv(full_path, mode='a',
              header=False, index=False) 
    df = pd.DataFrame(data=filepath_list_multiclass)
    df.to_csv(full_path, mode='a',
              header=False, index=False)  
    return


def rand_card_position(win, pos_all, class_strlist_english, class_strlist_chinese,
                       img_class):
    card_list = []
    text_list = []
    card_vert = [[0.15, 0.1], [-0.15, 0.1], [-0.15, -0.1], [0.15, -0.1]]
    region_color = 'white'
    text_color = 'white'
    text_hight = 0.08
    opacity_ori = 0.3  
    class_num = len(class_strlist_english)
    class_number = []
    for index in range(0, class_num):
        if index!=img_class:
            class_number.append(index)
    choice_show_num = 10  
    show_class=random.sample(population=class_number, k=choice_show_num-1)
    show_class.append(img_class)
    random.shuffle(show_class)
    for posNO,classNO in enumerate(show_class):
        class_str_english = class_strlist_english[classNO]
        class_str_chinese = class_strlist_chinese[classNO]
        thisclass_card = visual.ShapeStim(win, lineColor=None,
                                          fillColor=region_color,
                                          opacity=opacity_ori,
                                          vertices=card_vert,
                                          pos=pos_all[posNO])
        thisclass_text = visual.TextStim(win, text=class_str_chinese+'\n'+class_str_english,
                                         height=text_hight,
                                         color=text_color,
                                         pos=pos_all[posNO])     
        card_list.append(thisclass_card)  
        text_list.append(thisclass_text)
    none_card = visual.ShapeStim(win, lineColor=None,
                                 fillColor=region_color,
                                 opacity=opacity_ori,
                                 vertices=card_vert,
                                 pos=pos_all[choice_show_num]) 
    none_text = visual.TextStim(win, text='无\nnone',
                                height=text_hight,
                                color=text_color,
                                pos=pos_all[choice_show_num])  
    card_list.append(none_card)
    text_list.append(none_text)
    return card_list, text_list

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


def select_serialwrite(win, myMouse, card_list, img_class_str, text_list, ser):
    class_num = len(card_list)-1  
    opacity_ori = 0.3  
    opacity_mouse_touch = 0.4  
    while (True): 
        for cardNO, this_card in enumerate(card_list):
            if myMouse.isPressedIn(this_card):
                choice_class_str = text_list[cardNO].text.split('\n')[-1]
                flag_correct = is_correct_feedback(
                    choice_class_str, img_class_str, class_num, ser) 
                return choice_class_str, flag_correct
            if this_card.contains(myMouse):
                this_card.opacity = opacity_mouse_touch
            else:
                this_card.opacity = opacity_ori
            this_card.draw()
            this_text = text_list[cardNO]
            this_text.draw()
        win.flip()


def is_correct_feedback(choice_class_str, img_class_str, class_num, ser):
    if choice_class_str == img_class_str:
        ser.write([0])
        ser.write([252])  # means correct
        for i in range(1,20):
            ser.write([0])
        flag_correct = True
        # pyttsx3.speak('right')
    elif choice_class_str == 'none': 
        ser.write([0])
        ser.write([254])  # means wrong 
        for i in range(1,20):
            ser.write([0])
        flag_correct = False
        # pyttsx3.speak('wrong')
    else:
        ser.write([0])
        ser.write([254])  # means wrong
        for i in range(1,20):
            ser.write([0])
        flag_correct = False
        pyttsx3.speak('wrong')
    return flag_correct


def save_exp_info(exp_info_savepath, exp_info_filename, save_list_row):
    full_path = os.path.join(exp_info_savepath, exp_info_filename)
    if not os.path.exists(full_path):
        columns = ['imgNO', 'img_path',
                   'class_str', 'img_class',   'imgshow_time',   'imgshow_localtime_str',
                   'choice_class_str',   'choice_class',   'select_time',   'select_localtime_str',
                   'flag_correct', 'right_number']
        df = pd.DataFrame(columns=columns)
        df.to_csv(full_path, index=False)
    df = pd.DataFrame(data=[save_list_row]) 
    df.to_csv(full_path, mode='a',
              header=False, index=False)  
    return
