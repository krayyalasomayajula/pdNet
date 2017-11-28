import sys
import argparse
import math

import cv2
import pdb

import os
from os import listdir
from os.path import isfile, join
import numpy as np

def pngImgDirs(x_args):
    #pdb.set_trace()
    in_path     = os.path.abspath(x_args.indir)
    out_path    = os.path.abspath(x_args.outdir)
    
    out_path    = join(out_path,x_args.opt)
    
    oDirs   = [join(out_path,'img'), join(out_path,'gt')]
    iDirs   = [join(in_path,'img'), join(in_path,'gt')]
    
    for od in oDirs: 
        try:
            os.stat(od)
        except:
            os.makedirs(od)
    
    onlyfiles   = []
    
    for fil in listdir(iDirs[0]):
        if(isfile(join(iDirs[0],fil)) and isfile(join(iDirs[1],'binaryresult_'+fil))):
                onlyfiles.append(fil)
        else:
            print("GT missing for file %s" % (fil))
                        
    #pdb.set_trace()
    onlyfiles = sorted(onlyfiles)
    for fil in onlyfiles:
        print('Proocessing file: %s' % fil)

        imgFile   = join(iDirs[0],fil)
        gtFile    = join(iDirs[1],'binaryresult_'+fil)
        
        fName = fil.split('.')[0]+'.png'
        
        ximg = cv2.imread(imgFile)
        if(len(ximg.shape) > 2):
            img = cv2.cvtColor(ximg [:,:,:3], cv2.COLOR_BGR2GRAY)
        else:
            img = ximg
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(join(oDirs[0],fName),img)
        
        ximg = cv2.imread(gtFile) #GT file
        if(len(ximg.shape) > 2):
            img = cv2.cvtColor(ximg [:,:,:3], cv2.COLOR_BGR2GRAY)
        else:
            img = ximg
        img = (255.0*(img > 254)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        cv2.imwrite(join(oDirs[1],fName),img)
    
'''
Usage:
python bickleyImgPNGs.py -i ../../imgDataSets/bickleyDairy/orig/train -o ../../imgDataSets/bickleyDairy/pngs -opt train
python bickleyImgPNGs.py -i ../../imgDataSets/bickleyDairy/orig/test -o ../../imgDataSets/bickleyDairy/pngs -opt test
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str, default=[''])
    parser.add_argument("-o", "--outdir", type=str, default=[''])
    parser.add_argument("-opt", type=str, default=['train'])
    
    args = parser.parse_args()
    
    pngImgDirs(args)
    