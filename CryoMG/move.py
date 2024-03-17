import os
import pandas as pd
import shutil

path = r'H:\mmcode_GryoGM\mmclassification-master\mmcls\data\GryoGM'
label = pd.read_csv('H:\mmcode_GryoGM\mmclassification-master\mmcls\data\label.csv',header=-1,index_col=0)
label.index.rename(None, inplace = True)
label.columns = ['label']

for root, dirs, files in os.walk(path):
    for file in files:
        if file[-4:]=='.jpg':
            ppp = 'H:\\mmcode_GryoGM\\mmclassification-master\\mmcls\\data\\train\\'+str(label.loc[file]['label'])
            shutil.copy(os.path.join(root,file), os.path.join(ppp,file))

        