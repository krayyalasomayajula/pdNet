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
        
    
    in_datapath     = os.path.abspath(xargs.indir)
    out_datapath    = os.path.abspath(xargs.outdir)
    cue_datapath    = os.path.abspath(xargs.cuedir)
    
    stride  = xargs.step
    ph      = xargs.ph
    pw      = xargs.pw
    
    fmt = Formatter()
    
    try:
        os.stat(out_datapath)
    except:
        os.mkdir(out_datapath)       
    
    onlyfiles = sorted([f for f in listdir(cue_datapath) if isfile(join(cue_datapath, f))])
    
    #pdb.set_trace()
    
    for fil in onlyfiles:
        print(" ====== Processing: %s ======" % fil)
        
        cue_fileName = os.path.join(cue_datapath,fil)
        cue_img      = cv2.imread(cue_fileName)
        out_img      = 0 * cue_img
        
        patch_idx = 1
        fname, ext = fil.split('.')
        
        patch_fileName = os.path.join(in_datapath,fname+'_'+fmt.format("%04d" % patch_idx)+'.'+ext)
        patch_ta    = cv2.imread(patch_fileName)
        
        #pdb.set_trace()
        
        if(ph < 0 and pw < 0):
            ph          = patch_ta.shape[0]
            pw          = patch_ta.shape[1]
            stride      = 0.75*ph
        
        hstride = float(stride)
        wstride = (pw*float(stride)/ph)
        
        h_strides = int(np.ceil((cue_img.shape[0]-ph)/hstride))
        w_strides = int(np.ceil((cue_img.shape[1]-pw)/wstride))
                    
        hrange = range(0, h_strides*int(hstride), int(hstride))
        wrange = range(0, w_strides*int(wstride), int(wstride))

      # collect patches_ta
        for h in hrange:
            for w in wrange:
                patch_fileName = os.path.join(in_datapath,fname+'_'+fmt.format("%04d" % patch_idx)+'.'+ext)
                patch_ta    = cv2.imread(patch_fileName)
                patch_idx += 1
                if(len(out_img.shape) > 2):
                    out_img[h : h + ph, w : w + pw, :]  = patch_ta
                else:
                    out_img[h : h + ph, w : w + pw]     = patch_ta
                
                #print("file %s: idx:%d" % (fil,patch_idx))
                #pdb.set_trace()
                
            
            # Column boundary patches
            patch_fileName = os.path.join(in_datapath,fname+'_'+fmt.format("%04d" % patch_idx)+'.'+ext)
            patch_ta    = cv2.imread(patch_fileName)
            patch_idx += 1
            if(len(out_img.shape) > 2):
                out_img[h : h + ph, -pw:, :]    = patch_ta
            else:
                out_img[h : h + ph, -pw:]       = patch_ta
        
        # Row boundary patches
        for w in wrange:
            # print(h, w, out_img.shape[2], out_img.shape[3])
            patch_fileName = os.path.join(in_datapath,fname+'_'+fmt.format("%04d" % patch_idx)+'.'+ext)
            patch_ta    = cv2.imread(patch_fileName)
            patch_idx += 1
            
            if(len(out_img.shape) > 2):
                out_img[-ph:, w : w + pw, :]    = patch_ta
            else:
                out_img[-ph:, w : w + pw]       = patch_ta
        
        patch_fileName = os.path.join(in_datapath,fname+'_'+fmt.format("%04d" % patch_idx)+'.'+ext)
        patch_ta    = cv2.imread(patch_fileName)
        patch_idx += 1
        
        if(len(out_img.shape) > 2):
            out_img[-ph:, -pw:, :]  = patch_ta
        else:
            out_img[-ph:, -pw:]     = patch_ta
        
        if(xargs.verify):
            err = np.sum(np.abs(cue_img - out_img))
            if(err):
                print('Cummulative error for %s file is: %d' % (fil,err))
        
        out_fileName = os.path.join(out_datapath,fil)
        cv2.imwrite(out_fileName,out_img)
        
        #pdb.set_trace()

'''
Usage:
python dibcoImgJoin.py
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str, default='../datasets/xDIBCO/test/')
    parser.add_argument("-o", "--outdir", type=str, default='../datasets/jDIBCO/img/')
    parser.add_argument("-c", "--cuedir", type=str, default='../../imgDataSets/DIBCO/pngs/img')
    #parser.add_argument("-ph", type=int, default=200)
    #parser.add_argument("-pw", type=int, default=256)
    parser.add_argument("-verify", type=int, default=0)
    
    args = parser.parse_args()
    
    args.step   = 0.0
    args.ph     = -1
    args.pw     = -1
    
    getImgTiles(args)
    
