import sys
import argparse
import math

import cv2
import pdb

import os
from os import listdir
from os.path import isfile, join

def pngImgDirs(x_args):
    #pdb.set_trace()
    in_path     = os.path.abspath(x_args.indir)
    out_path    = os.path.abspath(x_args.outdir)
    years       = listdir(in_path)
    years_dirs  = [join(in_path,dir) for dir in years]
    onlyfiles   = {'img':[], 'gt':[]}
    
    for y_dir in years_dirs:
        #print(y_dir)
        for img_gt in listdir(y_dir):
            for fil in listdir(join(y_dir,img_gt)):
                if(isfile(join(y_dir,img_gt,fil))):
                    onlyfiles[img_gt].append(join(y_dir,img_gt,fil))
    
    onlyfiles['img'] = sorted(onlyfiles['img'])
    onlyfiles['gt'] = sorted(onlyfiles['gt'])
    
    #pdb.set_trace()
    for dir, fil_list in onlyfiles.iteritems():
        outdir = join(out_path,dir)
        #print(outdir)
        try:
            os.stat(outdir)
        except:
            os.mkdir(outdir)       
        
        for fil in fil_list:
            ximg = cv2.imread(fil)
            if(len(ximg.shape) > 2):
                img = ximg [:,:,:3]
            else:
                img = ximg
                
            yr_fname = fil.split('/')[-3]+'_'+(fil.split('/')[-1]).split('.')[0]+'.png'
            print(join(outdir,yr_fname))
            cv2.imwrite(join(outdir,yr_fname),img)

'''
Usage:
python dibcoImgPNGs.py -i ../datasets/DIBCO/original -o ../datasets/DIBCO/pngs
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str, default=[''])
    parser.add_argument("-o", "--outdir", type=str, default=[''])
    
    args = parser.parse_args()
    
    pngImgDirs(args)
    