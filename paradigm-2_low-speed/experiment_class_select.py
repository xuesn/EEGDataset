from __future__ import division

from psychopy import visual, core, event
from psychopy.hardware import keyboard

import time

import serial

from utils_class_select import rand_sample_multiclass, rand_card_position
from utils_class_select import imgshow_serialwrite, select_serialwrite
from utils_class_select import save_filepath_per_experiment, save_exp_info

exp_dir_path = 'C:\\Users\\Desktop\\eeg'
pic_path = exp_dir_path + '\\pic_10000_resized\\'  

sample_num_per_class = 5
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
              'none': 98, 'None': 98, 'NONE': 98,
              '飞机': 0, '自行车': 1, '鸟': 2, '船': 3, '瓶子': 4,
              '巴士': 5, '汽车': 6, '猫': 7, '椅子': 8, '牛': 9,
              '餐桌': 10, '狗': 11, '花': 12, '马': 13, '助力车': 14,
              '人': 15, '羊': 16, '沙发': 17, '火车': 18, '显示器': 19,
              'special': 99}
class_num = len(class_strlist_english)

img_list = rand_sample_multiclass(
    pic_path, sample_num_per_class, class_strlist_english)


#
##
###--SPECIAL--###
exp_date = ''
sub_name = ''  
expNO =   
###--SPECIAL--###
##
#

piclist_savepath = exp_dir_path+'\\paradigm-2_low-speed\\img_info_Class_Select\\'
filepath_save_name = exp_date+sub_name + \
    str(expNO)+'_sample'+str(sample_num_per_class) + \
    'class'+str(class_num)+'_Class_Select.csv'
save_filepath_per_experiment(piclist_savepath, filepath_save_name, img_list)
# end----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
fresh_rate = 60
singleobj_time_imgshow = 1  # picture present time
singleobj_frameN_imgshow = singleobj_time_imgshow * fresh_rate
singleobj_time_blank_afterimg = 0  # picture present  ISI time
singleobj_frameN_blank_afterimg = singleobj_time_blank_afterimg * fresh_rate
# after select time before next picture present
singleobj_time_blank_afterselect = 1
singleobj_frameN_blank_afterselect = singleobj_time_blank_afterselect * fresh_rate
x = [-0.35, 0, 0.35, 0.7]
y = [0.375, 0.125, -0.125, -0.375]
pos_all = [[x[0], y[0]], [x[1], y[0]], [x[2], y[0]],
           [x[0], y[1]],               [x[2], y[1]],
           [x[0], y[2]],               [x[2], y[2]],
           [x[0], y[3]], [x[1], y[3]], [x[2], y[3]],  [x[3], y[3]]]
# end----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
ser = serial.Serial('COM4', 115200, timeout=1)
for i in range(1,20):
    ser.write([0])
win = visual.Window(size=(1920, 1080), fullscr=False, screen=0, allowGUI=False, allowStencil=False,
                    monitor='testMonitor', color=[-1, -1, -1], colorSpace='rgb', blendMode='avg', useFBO=True)
# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()
myMouse = event.Mouse()
pic_all_win = []
for i in range(len(img_list)):
    pic_all_win.append(visual.ImageStim(win, image=img_list[i], mask=None, units='', pos=(0.0, 0.0), size=None, ori=0.0, color=(1.0, 1.0, 1.0),
                                        colorSpace='rgb', contrast=1.0, opacity=1.0, depth=0, interpolate=False, flipHoriz=False, flipVert=False, texRes=128, name=None, autoLog=None, maskParams=None))
# end----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
intro = visual.TextStim(win, text=u'按空格开始', height=0.12,
                        pos=(-0.1, 0.1), bold=True, italic=False, color='white')
intro.draw()
win.flip()
k_1 = event.waitKeys(keyList=['space'])
print(k_1)

exp_info_savepath = exp_dir_path + \
    '\\paradigm-2_low-speed\\exp_info_Class_Select\\'
exp_info_filename = exp_date+sub_name+str(expNO)+'_exp_info_Class_Select.csv'
exp_startNO = 1
exp_breakNO = 1
save_list_row_start = ['experiment-start!', time.time(), time.asctime(
    time.localtime(time.time()))]  
save_exp_info(exp_info_savepath, exp_info_filename, save_list_row_start)
exp_startNO += 1

for frameN in range(singleobj_frameN_blank_afterimg):
    win.flip()

right_number = 0  
img_num_per_break = 20  
selection_num = len(img_list)
for imgNO in range(1, selection_num+1):
    # --------------------------------------------------
    print('\n####imageNO-'+str(imgNO)+'####')
    pic_win = pic_all_win[imgNO-1]

    imgshow_time = time.time()
    imgshow_localtime_str = time.asctime(time.localtime(time.time()))
    imgshow_serialwrite(
        pic_win, imgNO, win, singleobj_frameN_imgshow, singleobj_frameN_blank_afterimg, ser)

    img_path = img_list[imgNO-1]
    path_split_str = '\\'  
    img_class_str = img_path.split(path_split_str)[-2]  
    img_class = class_dict[img_class_str]  
    print('        ', img_class_str)

    # --------------------------------------------------
    card_list, text_list = rand_card_position(
        win, pos_all, class_strlist_english, class_strlist_chinese,
        img_class)  
    choice_class_str, flag_correct = select_serialwrite(
        win, myMouse, card_list, img_class_str, text_list, ser)
    choice_class = class_dict[choice_class_str]
    print('SELECT: ', choice_class_str)
    select_time = time.time()
    select_localtime_str = time.asctime(time.localtime(time.time()))
    right_number += int(flag_correct)  
    
    for frameN in range(singleobj_frameN_blank_afterselect):
        white_cross = visual.TextStim(win, text=u'+', height=0.20,
                                      pos=(0, 0), bold=True, italic=False, color='white')
        white_cross.draw()
        win.flip()

    save_list_row_imgshow_select = [imgNO, img_path,
                                    img_class_str, img_class, imgshow_time, imgshow_localtime_str,
                                    choice_class_str, choice_class,   select_time,  select_localtime_str,
                                    str(flag_correct),   right_number]
    save_exp_info(exp_info_savepath, exp_info_filename,
                  save_list_row_imgshow_select)

    # --------------------------------------------------
    if (imgNO) % img_num_per_break == 0:
        print('\nBREAK-TIME')

        save_list_row_break = ['experiment-break-time!', time.time(), time.asctime(
            time.localtime(time.time()))]  
        save_exp_info(exp_info_savepath, exp_info_filename,
                      save_list_row_break)
        exp_breakNO += 1

        # winsound.Beep(2000, 500)
        intro = visual.TextStim(win, text=u'请休息', height=0.12,
                                pos=(-0.1, 0.1), bold=True, italic=False, color='white')
        intro.draw()
        win.flip()
        time.sleep(5)  
        # winsound.Beep(1000, 500)
        intro = visual.TextStim(win, text=u'按空格开始', height=0.12,
                                pos=(-0.1, 0.1), bold=True, italic=False, color='white')
        intro.draw()
        win.flip()
        k_1 = event.waitKeys(keyList=['space'])  

        save_list_row_start = ['experiment-start!', time.time(), time.asctime(
            time.localtime(time.time()))]  
        save_exp_info(exp_info_savepath, exp_info_filename,
                      save_list_row_start)
        exp_startNO += 1

        for frameN in range(singleobj_time_blank_afterselect):
            win.flip()

    # check for quit (typically the Esc key)
    if defaultKeyboard.getKeys(keyList=["escape"]):
        print('\nQUIT-GAME')
        ser.close()
        win.close()
        core.quit()
# end----------------------------------------------------------------------------------------------------

print(f"correct rate: {right_number/(imgNO)}")
print('\nQUIT-GAME')
ser.close()
win.close()
core.quit()
