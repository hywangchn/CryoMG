<<<<<<< HEAD
import os
import numpy as np
import shutil


path = r'H:\mmcode_GryoGM\mmclassification-master\mmcls\data\train_backup'
new_path = r'H:\mmcode_GryoGM\mmclassification-master\mmcls\data\data'

i=0
for root, dirs, files in os.walk(path):
    for file in files:
        if i%3==0:
            temp_path = new_path+'\\train\\'+root[-1]
            shutil.copy(os.path.join(root, file), os.path.join(temp_path, file))
        elif i%3==1:
            temp_path = new_path+'\\val\\'+root[-1]
            shutil.copy(os.path.join(root, file), os.path.join(temp_path, file))
        else:
            temp_path = new_path+'\\test\\'+root[-1]
            shutil.copy(os.path.join(root, file), os.path.join(temp_path, file))
=======
import os
import numpy as np
import shutil


path = r'H:\mmcode_GryoGM\mmclassification-master\mmcls\data\train_backup'
new_path = r'H:\mmcode_GryoGM\mmclassification-master\mmcls\data\data'

i=0
for root, dirs, files in os.walk(path):
    for file in files:
        if i%3==0:
            temp_path = new_path+'\\train\\'+root[-1]
            shutil.copy(os.path.join(root, file), os.path.join(temp_path, file))
        elif i%3==1:
            temp_path = new_path+'\\val\\'+root[-1]
            shutil.copy(os.path.join(root, file), os.path.join(temp_path, file))
        else:
            temp_path = new_path+'\\test\\'+root[-1]
            shutil.copy(os.path.join(root, file), os.path.join(temp_path, file))
>>>>>>> c2a53a595f4c57c357d58c4c764e7338c4532a05
        i+=1