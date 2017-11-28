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
    
    if(x_args.opt == 'test'):
        eval_path   = os.path.abspath(x_args.evaldir)
    
    out_path    = join(out_path,x_args.opt)
    
    oDirs   = [join(out_path,'img'), join(out_path,'gt')]
    
    for od in oDirs: 
        try:
            os.stat(od)
        except:
            os.makedirs(od)
    
    onlyfiles   = []
    gtdir = ''
    
    for fil in listdir(in_path):
        fName, fExt = fil.split('.')[0], fil.split('.')[1]
        if(fExt == 'jpg'):
            if(isfile(join(in_path,fil))):
                
                if(x_args.opt == 'train'):
                    gtdir = in_path
                if(x_args.opt == 'test'):
                    gtdir = eval_path
                                          
                if(isfile(join(gtdir,fName+'_GT1.bmp')) and isfile(join(gtdir,fName+'_GT2.bmp'))):
                    onlyfiles.append(fil)
                else:
                    print("GT1 or GT2 missing for file %s" % (fil))
                        
    #pdb.set_trace()
    onlyfiles = sorted(onlyfiles)
    for fil in onlyfiles:
        print('Proocessing file: %s' % fil)
        
        fName  = fil.split('.')[0]
        iName1 = fName+'_GT1.png' #Creating two inputs for two GTs 
        iName2 = fName+'_GT2.png'
        
        oName1 = fName+'_GT1.png' #Two types of GTs available 
        oName2 = fName+'_GT2.png'
        
        inFile      = join(in_path,fil)
        
        outFile1    = join(gtdir,fName+'_GT1.bmp')
        outFile2    = join(gtdir,fName+'_GT2.bmp')
        
        ximg = cv2.imread(inFile)
        if(len(ximg.shape) > 2):
            img = ximg [:,:,:3]
        else:
            img = ximg
        
        cv2.imwrite(join(oDirs[0],iName1),img) # Create 2 inputs
        cv2.imwrite(join(oDirs[0],iName2),img)
        
        ximg = cv2.imread(outFile1) #Handle GT1
        if(len(ximg.shape) > 2):
            img = cv2.cvtColor(ximg[:,:,:3], cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(ximg, cv2.COLOR_BGR2GRAY)
            
        cv2.imwrite(join(oDirs[1],oName1),img)
        
        ximg = cv2.imread(outFile2) #Handle GTs
        if(len(ximg.shape) > 2):
            img = cv2.cvtColor(ximg[:,:,:3], cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(ximg, cv2.COLOR_BGR2GRAY)
            
        cv2.imwrite(join(oDirs[1],oName2),img)

'''
Usage:
python balineseImgPNGs.py -i ../../imgDataSets/balinese/Challenge-1-ForTrain/train-50 -o ../../imgDataSets/balinese/pngs -opt train
python balineseImgPNGs.py -i ../../imgDataSets/balinese/Challenge-1-ForTest/test-50 -o ../../imgDataSets/balinese/pngs -e ../../imgDataSets/balinese/Challenge-1-ForEvaluation/test-50-GT -opt test
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str, default=[''])
    parser.add_argument("-o", "--outdir", type=str, default=[''])
    parser.add_argument("-opt", type=str, default=['train'])
    parser.add_argument("-e", "--evaldir", type=str, default=['..'])
    
    
    args = parser.parse_args()
    
    pngImgDirs(args)
    