<<<<<<< HEAD
import os
from PIL import Image

path = r'./data/CryoGM'

for root, dirs, files in os.walk(path):
    for file in files:
        if file[-4:]=='.jpg':
            file_path = os.path.join(root,file)
            img = Image.open(file_path)
            img = img.resize((256,256),Image.ANTIALIAS)
            temp = ''
            temp_list = root.split('\\')
            for i in range(len(temp_list)):
                if i>=2:
                    temp = temp+temp_list[i]+'/'
                    if not os.path.exists('./GryoGM_down/'+temp):
                        os.mkdir('./cryoppp_down/'+temp)
            new_path = os.path.join('./cryoppp_down/'+temp,file)
=======
import os
from PIL import Image

path = r'./data/CryoGM'

for root, dirs, files in os.walk(path):
    for file in files:
        if file[-4:]=='.jpg':
            file_path = os.path.join(root,file)
            img = Image.open(file_path)
            img = img.resize((256,256),Image.ANTIALIAS)
            temp = ''
            temp_list = root.split('\\')
            for i in range(len(temp_list)):
                if i>=2:
                    temp = temp+temp_list[i]+'/'
                    if not os.path.exists('./GryoGM_down/'+temp):
                        os.mkdir('./cryoppp_down/'+temp)
            new_path = os.path.join('./cryoppp_down/'+temp,file)
>>>>>>> c2a53a595f4c57c357d58c4c764e7338c4532a05
            img.save(new_path,quality=100)