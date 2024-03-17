import os
import pandas as pd

path = r'./data/'

result = pd.DataFrame(columns=['Name','Label'])

for root, dirs, files in os.walk(path):
    if len(root)==12:
        record = pd.DataFrame(columns=['Name','Label'])
        micrographs = os.listdir(os.path.join(root, 'micrographs'))
        ground_truth = os.listdir(os.path.join(root,'ground_truth/particle_coordinates'))
        for i in range(len(micrographs)):
            micrographs[i] = micrographs[i][:-4]
        for i in range(len(ground_truth)):
            ground_truth[i] = ground_truth[i][:-4]
            
            
        for i in range(len(micrographs)):
            if micrographs[i] in ground_truth:
                record.loc[i] = [micrographs[i]+'.jpg','1']
            else:
                record.loc[i] = [micrographs[i]+'.jpg','0']
        #record.to_csv(os.path.join(root,'record.csv'), header=None, index=None)
        result = pd.concat([result,record], axis=0)
result.to_csv(os.path.join(path,'record.csv'), header=None, index=None)