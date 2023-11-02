# -*- coding: utf-8 -*
import os
from os import listdir

'''
Generate the file path of dataset.
the last null string should be deleted manually.
'''
img_path = '../SEN12MS_CR_SARColorData/sar_train/'
img_list=os.listdir(img_path)

with open('sar_train_filepath_list.txt','w') as f:
    for img_name in img_list:
        f.write("\'" + img_path + img_name + "\'" + ",")
