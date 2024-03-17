# -*- coding: utf-8 -*-

import os
import cv2

load_path = 'F:\\code\\transferlearning\\train\\'
record_path = 'F:\\code\\mmcode\\mmclassification-master\\mmcls\\data\\train'



for root, dirs, files in os.walk(load_path):
    r_path = record_path+'\\'+root.split('\\')[-1]
    if not os.path.exists(r_path):
        os.makedirs(r_path)
    for file in files:
        img = cv2.imread(os.path.join(root, file))
        img = cv2.resize(img,(500,500))
        cv2.imwrite(os.path.join(r_path,file),img)