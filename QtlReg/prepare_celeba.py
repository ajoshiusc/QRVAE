import random
import pickle
import zipfile

import numpy as np

from scipy import misc
import imageio
import tqdm

from dlutils import download
from PIL import Image


corrupted = [
    '195995.jpg',
    '131065.jpg',
    '118355.jpg',
    '080480.jpg',
    '039459.jpg',
    '153323.jpg',
    '011793.jpg',
    '156817.jpg',
    '121050.jpg',
    '198603.jpg',
    '041897.jpg',
    '131899.jpg',
    '048286.jpg',
    '179577.jpg',
    '024184.jpg',
    '016530.jpg',
]

download.from_google_drive("0B7EVK8r0v71pZjFTYXZWM3FlRnM")


def center_crop(x, crop_h=128, crop_w=None, resize_w=64):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h # the width and height after cropped
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.)) + 15
    i = int(round((w - crop_w)/2.))
    im = Image.fromarray(x[j:j+crop_h, i:i+crop_w])
    return np.array(im.resize([resize_w, resize_w]))


archive = zipfile.ZipFile('img_align_celeba.zip', 'r')

names = archive.namelist()

names = [x for x in names if x[-4:] == '.jpg']

count = len(names)
print("Count: %d" % count)

names = [x for x in names if x[-10:] not in corrupted]

folds = 5

random.shuffle(names)

images = {}

count = len(names)
print("Count: %d" % count)
count_per_fold = count // folds

i = -1
im = 0
valid=1
for x in tqdm.tqdm(names):
    imgfile = archive.open(x)
    image = center_crop(imageio.imread(imgfile))
    images[x] = image
    im += 1


    if (im ==128 and valid==1) :
        output = open('data_valid.pkl', 'wb')
        pickle.dump(list(images.values()), output)
        output.close()
        i += 1
        im = 0
        valid=0
        images.clear()
        
    elif im == count_per_fold:
        output = open('data_v2_fold_%d.pkl' % i, 'wb')
        pickle.dump(list(images.values()), output)
        output.close()
        i += 1
        im = 0
        images.clear()
