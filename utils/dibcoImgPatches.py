import argparse
import os
import sys
from os import listdir
from os.path import isfile, join

import pdb
import cv2
import numpy as np
from string import Formatter


def getImgTiles(xargs):
    
    #pdb.set_trace()
    in_dirs     = ['gt','img']
    in_datapath = os.path.abspath(xargs.indir)
    
    if(xargs.train):
	    out_dirs     = ['trainannot','train']
    if(xargs.valid):
	    out_dirs     = ['validannot','valid']
    
    out_datapath = os.path.abspath(xargs.outdir+'/'+str(xargs.ph)+'x'+str(xargs.pw))
    
    stride  = xargs.step
    ph      = xargs.ph
    pw      = xargs.pw
    
    fmt = Formatter()
    for i_dir, o_dir in zip(in_dirs, out_dirs):
        in_dir  = os.path.join(in_datapath,i_dir)
        out_dir = os.path.join(out_datapath,o_dir)
        
        try:
            os.stat(out_dir)
        except:
            os.makedirs(out_dir)       
        
        onlyfiles = sorted([f for f in listdir(in_dir) if isfile(join(in_dir, f))])
        
        #pdb.set_trace()
        
        for fil in onlyfiles:
            print(" ====== Processing: %s ======" % fil)
            
            in_fileName = os.path.join(in_dir,fil)
            img = cv2.imread(in_fileName)
            
            if(xargs.gray):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if(img.shape[0] < ph or img.shape[1] < pw):
        		print("dimensions of file %s are smaller" % (in_fileName))
        		continue

            if(i_dir == 'img' and len(img.shape) == 2):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            hstride = float(stride)
            wstride = (pw*float(stride)/ph)
            
            h_strides = int(np.ceil((img.shape[0]-ph)/hstride))
            w_strides = int(np.ceil((img.shape[1]-pw)/wstride))
                        
            hrange = range(0, h_strides*int(hstride), int(hstride))
            wrange = range(0, w_strides*int(wstride), int(wstride))

          # collect patches_ta
            patch_idx = 1
            fname, ext = fil.split('.')
            
            for h in hrange:
                for w in wrange:
                    # print(h, w, img.shape[2], img.shape[3])
                    if(len(img.shape) > 2):
                        patch_ta = img[h : h + ph, w : w + pw, :]
                    else:
                        patch_ta = img[h : h + ph, w : w + pw]
                    
                    #print("file %s: idx:%d" % (fil,patch_idx))
                    #pdb.set_trace()
                    
                    out_fileName = os.path.join(out_dir,fname+'_'+fmt.format("%04d" % patch_idx)+'.'+ext)
                    cv2.imwrite(out_fileName,patch_ta)
                    patch_idx += 1
                
                # Column boundary patches
                if(len(img.shape) > 2):
                    patch_ta = img[h : h + ph, -pw:, :]
                else:
                    patch_ta = img[h : h + ph, -pw:]
                
                out_fileName = os.path.join(out_dir,fname+'_'+fmt.format("%04d" % patch_idx)+'.'+ext)
                cv2.imwrite(out_fileName,patch_ta)
                patch_idx += 1
            
            # Row boundary patches
            for w in wrange:
                # print(h, w, img.shape[2], img.shape[3])
                if(len(img.shape) > 2):
                    patch_ta = img[-ph:, w : w + pw, :]
                else:
                    patch_ta = img[-ph:, w : w + pw]
                
                out_fileName = os.path.join(out_dir,fname+'_'+fmt.format("%04d" % patch_idx)+'.'+ext)
                cv2.imwrite(out_fileName,patch_ta)
                patch_idx += 1
            
            if(len(img.shape) > 2):
                patch_ta = img[-ph:, -pw:, :]
            else:
                patch_ta = img[-ph:, -pw:]
            
            out_fileName = os.path.join(out_dir,fname+'_'+fmt.format("%04d" % patch_idx)+'.'+ext)
            cv2.imwrite(out_fileName,patch_ta)
            patch_idx += 1
            
            #pdb.set_trace()

'''
Usage: 
>>>>>>>>>>>> Color Datasets
-- DIBCO datasets --
python dibcoImgPatches.py -i ../../imgDataSets/DIBCO/augmented -o ../datasets/xDibco -ph 128 -pw 256 -train 1
python dibcoImgPatches.py -i ../../imgDataSets/DIBCO/pngs -o ../datasets/xDibco -ph 128 -pw 256 -valid 1
-- Synthetic datasets --
python dibcoImgPatches.py -i ../../imgDataSets/Synthetic/augmented -o ../datasets/xSynthetic -ph 128 -pw 256 -train 1
python dibcoImgPatches.py -i ../../imgDataSets/Synthetic/original -o ../datasets/xSynthetic -ph 128 -pw 256 -valid 1

>>>>>>>>>>>> Grayscale Datasets
cd ../utils;python dibcoImgPatches.py -i ../../imgDataSets/DIBCO/augmented -o ../datasets/xDibcoGray -ph 128 -pw 256 -train 1 -gray 1

>>>>>>>>>>>> Balinese Datasets
python dibcoImgPatches.py -i ../../imgDataSets/balinese/augmented/train -o ../datasets/xBalinese -ph 128 -pw 256 -train 1
python dibcoImgPatches.py -i ../../imgDataSets/balinese/pngs/train -o ../datasets/xBalinese -ph 128 -pw 256 -valid 1
-- Rename valid to valid_ and validannot to validannot_
mv ../datasets/xBalinese/128x256/valid ../datasets/xBalinese/128x256/valid_
mv ../datasets/xBalinese/128x256/validannot ../datasets/xBalinese/128x256/validannot_
python dibcoImgPatches.py -i ../../imgDataSets/balinese/pngs/test -o ../datasets/xBalinese -ph 128 -pw 256 -valid 1
-- Rename valid to test and validannot to testannot
-- Rename valid_ to valid and validannot_ to validannot
mv ../datasets/xBalinese/128x256/valid ../datasets/xBalinese/128x256/test
mv ../datasets/xBalinese/128x256/validannot ../datasets/xBalinese/128x256/testannot
mv ../datasets/xBalinese/128x256/valid_ ../datasets/xBalinese/128x256/valid
mv ../datasets/xBalinese/128x256/validannot_ ../datasets/xBalinese/128x256/validannot

python dibcoImgPatches.py -i ../../imgDataSets/balinese/augmented/train -o ../datasets/xBalineseGray -ph 128 -pw 256 -train 1 -gray 1
python dibcoImgPatches.py -i ../../imgDataSets/balinese/pngs/train -o ../datasets/xBalineseGray -ph 128 -pw 256 -valid 1 -gray 1
mv ../datasets/xBalineseGray/128x256/valid ../datasets/xBalineseGray/128x256/valid_
mv ../datasets/xBalineseGray/128x256/validannot ../datasets/xBalineseGray/128x256/validannot_
python dibcoImgPatches.py -i ../../imgDataSets/balinese/pngs/test -o ../datasets/xBalineseGray -ph 128 -pw 256 -valid 1  -gray 1
mv ../datasets/xBalineseGray/128x256/valid ../datasets/xBalineseGray/128x256/test
mv ../datasets/xBalineseGray/128x256/validannot ../datasets/xBalineseGray/128x256/testannot
mv ../datasets/xBalineseGray/128x256/valid_ ../datasets/xBalineseGray/128x256/valid
mv ../datasets/xBalineseGray/128x256/validannot_ ../datasets/xBalineseGray/128x256/validannot
>>>>>>>>>>>> Bickley Datasets
python dibcoImgPatches.py -i ../../imgDataSets/bickleyDairy/augmented/train -o ../datasets/xBickleyDairy -ph 128 -pw 256 -train 1
python dibcoImgPatches.py -i ../../imgDataSets/bickleyDairy/pngs/train -o ../datasets/xBickleyDairy -ph 128 -pw 256 -valid 1
mv ../datasets/xBickleyDairy/128x256/valid ../datasets/xBickleyDairy/128x256/valid_
mv ../datasets/xBickleyDairy/128x256/validannot ../datasets/xBickleyDairy/128x256/validannot_
python dibcoImgPatches.py -i ../../imgDataSets/bickleyDairy/pngs/test -o ../datasets/xBickleyDairy -ph 128 -pw 256 -valid 1
mv ../datasets/xBickleyDairy/128x256/valid ../datasets/xBickleyDairy/128x256/test
mv ../datasets/xBickleyDairy/128x256/validannot ../datasets/xBickleyDairy/128x256/testannot
mv ../datasets/xBickleyDairy/128x256/valid_ ../datasets/xBickleyDairy/128x256/valid
mv ../datasets/xBickleyDairy/128x256/validannot_ ../datasets/xBickleyDairy/128x256/validannot

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str, default='../datasets/DIBCO/augmented/')
    parser.add_argument("-o", "--outdir", type=str, default='../datasets/xDIBCO/')
    parser.add_argument("-ph", type=int, default=256)
    parser.add_argument("-pw", type=int, default=256)
    parser.add_argument("-train", type=int, default=0)
    parser.add_argument("-valid", type=int, default=0)
    parser.add_argument("-gray", type=int, default=0)

    args = parser.parse_args()
    
    if(args.valid and args.indir == '../datasets/DIBCO/augmented/'):
        args.indir = '../datasets/DIBCO/pngs/'
        print('Changing to default validation directory')
    
    args.step = 0.75*args.ph
    
    getImgTiles(args)
    
