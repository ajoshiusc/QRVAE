import glob
import numbers
import os
import shutil
import time
from functools import partial
from itertools import product, repeat
from multiprocessing import Pool, freeze_support
from shutil import copyfile

import numpy as np
import pandas as pd


def regparfun(subid, dirname='home'):

    print('dir=' + dirname + '  sub=' + subid)

    subdir = os.path.join(dirname, subid)
    subid = str(subid)
    print('reistering to mni ' + subid)

    t1 = os.path.join(subdir, subid + '_t1.nii.gz')
    t1mni = os.path.join(subdir, subid + '_t1mni.nii.gz')
    t1mnimask = os.path.join(subdir, subid + '_t1mni.mask.nii.gz')
    #    t1bfc = os.path.join(subdir, 'T1.bfc.nii.gz')

    t1mnimat = os.path.join(subdir, subid + '_t1mni.mat')
    print(subid)

    if not os.path.isfile(t1):
        print('T1 does not exist for ' + subid + t1)
        return

    # register T1 image to MNI space

    os.system(
        'flirt -in ' + t1 + ' -out ' + t1mni +
        ' -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -omat ' +
        t1mnimat + ' -dof 6 -cost normmi')

    # Create mask
    os.system('fslmaths ' + t1mni + ' -bin ' + t1mnimask)

    t1ce = os.path.join(subdir, subid + '_t1ce.nii.gz')
    t1cemni = os.path.join(subdir, subid + '_t1cemni.nii.gz')

    # register T1ce
    if os.path.isfile(t1ce):
        # Apply the same transform (T1->MNI) to register T1ce
        os.system('flirt -in ' + t1ce + ' -ref ' + t1mni + ' -out ' + t1cemni +
                  ' -applyxfm -init ' + t1mnimat)
        os.system('fslmaths ' + t1cemni + ' -mul ' + t1mnimask + ' ' + t1cemni)

    # register T2
    t2 = os.path.join(subdir, subid + '_t2.nii.gz')
    t2mni = os.path.join(subdir, subid + '_t2mni.nii.gz')
    if os.path.isfile(t2):
        # Apply the same transform (T1->MNI) to register T2
        os.system('flirt -in ' + t2 + ' -ref ' + t1mni + ' -out ' + t2mni +
                  ' -applyxfm -init ' + t1mnimat)
        os.system('fslmaths ' + t2mni + ' -mul ' + t1mnimask + ' ' + t2mni)

    # register Flair
    flair = os.path.join(subdir, subid + '_flair.nii.gz')
    flairmni = os.path.join(subdir, subid + '_flairmni.nii.gz')
    if os.path.isfile(flair):
        # Apply the same transform (T1->MNI) to register flair
        os.system('flirt -in ' + flair + ' -ref ' + t1mni + ' -out ' +
                  flairmni + ' -applyxfm -init ' + t1mnimat)
        os.system('fslmaths ' + flairmni + ' -mul ' + t1mnimask + ' ' +
                  flairmni)

    # register segmentations
    seg = os.path.join(subdir, subid + '_seg.nii.gz')
    segmni = os.path.join(subdir, subid + '_segmni.nii.gz')
    if os.path.isfile(seg):
        # Apply the same transform (T1->MNI) to registered segmentation
        os.system('flirt -in ' + seg + ' -ref ' + t1mni + ' -out ' + segmni +
                  ' -interp nearestneighbour -applyxfm -init ' + t1mnimat)
        os.system('fslmaths ' + segmni + ' -mul ' + t1mnimask + ' ' + segmni)

    return 0


def main():
    #Set subject dirs
    dir_name1 = '/ImagePTE1/ajoshi/BRATS2018/Training/HGG'
    subids1 = os.listdir(dir_name1)
    #    subids1 = [os.path.join(dir_name, file1) for file1 in subids1]

    dir_name2 = '/ImagePTE1/ajoshi/BRATS2018/Training/LGG'
    subids2 = os.listdir(dir_name2)
    #    subids2 = [os.path.join(dir_name, file1) for file1 in subids2]

    print(len(subids1), len(subids2))

    # regparfun(subids1[1], dir_name1)


    pool = Pool(processes=12)

    pool.map(partial(regparfun, dirname=dir_name1), subids1)
    pool.map(partial(regparfun, dirname=dir_name2), subids2)

    print('++++SUBMITTED++++++')

    pool.close()
    pool.join()


if __name__ == "__main__":
    freeze_support()
    main()
