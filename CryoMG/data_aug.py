<<<<<<< HEAD
import os
import numpy as np
from PIL import Image, ImageEnhance

def do_rotate(img, degree):
    img = img.rotate(degree)

    return img

def do_HorizontalFlip(img):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img

def do_EnhanceBrightness(img, degree):
    img_en = ImageEnhance.Brightness(img)
    img = img_en.enhance(degree)

    return img

def do_EnhanceContrast(img, degree):
    img_en = ImageEnhance.Contrast(img)
    img = img_en.enhance(degree)

    return img

def do_EnhanceSharpness(img, degree):
    img_en = ImageEnhance.Sharpness(img)
    img = img_en.enhance(degree)

    return img


path = './data/data/test/'

for root, dirs, files in os.walk(path):
    for file in files:
        if file[-4:]=='.jpg':
            img = Image.open(os.path.join(root, file))
            for degree in [45,90,135,180,225,270,315]:
                img_aug = do_rotate(img, degree)
                file_aug = file[:-4]+'rotate'+str(degree)+'.jpg'
                img_aug.save(os.path.join(root, file_aug))
            img_aug = do_HorizontalFlip(img)
            file_aug = file[:-4]+'HorizontalFlip'+'.jpg'
=======
import os
import numpy as np
from PIL import Image, ImageEnhance

def do_rotate(img, degree):
    img = img.rotate(degree)

    return img

def do_HorizontalFlip(img):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img

def do_EnhanceBrightness(img, degree):
    img_en = ImageEnhance.Brightness(img)
    img = img_en.enhance(degree)

    return img

def do_EnhanceContrast(img, degree):
    img_en = ImageEnhance.Contrast(img)
    img = img_en.enhance(degree)

    return img

def do_EnhanceSharpness(img, degree):
    img_en = ImageEnhance.Sharpness(img)
    img = img_en.enhance(degree)

    return img


path = './data/data/test/'

for root, dirs, files in os.walk(path):
    for file in files:
        if file[-4:]=='.jpg':
            img = Image.open(os.path.join(root, file))
            for degree in [45,90,135,180,225,270,315]:
                img_aug = do_rotate(img, degree)
                file_aug = file[:-4]+'rotate'+str(degree)+'.jpg'
                img_aug.save(os.path.join(root, file_aug))
            img_aug = do_HorizontalFlip(img)
            file_aug = file[:-4]+'HorizontalFlip'+'.jpg'
>>>>>>> c2a53a595f4c57c357d58c4c764e7338c4532a05
            img_aug.save(os.path.join(root, file_aug))