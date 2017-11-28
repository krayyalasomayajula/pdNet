import argparse
import os
import sys
from os import listdir
from os.path import isfile, join

import pdb
import cv2
import numpy as np
from string import Formatter

from scipy.io import loadmat, savemat

def getMatTiles(xargs):
    
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
        
        cue_fileName = os.path.join(cue_datapath,fil)
        cue_img      = cv2.imread(cue_fileName)
        
        patch_idx = 1
        fname, ext = fil.split('.')
        ext = 'mat' #change png to mat
        
        patch_fileName = os.path.join(in_datapath,fname+'_'+fmt.format("%04d" % patch_idx)+'.'+ext)
        patch_mat      = (loadmat(patch_fileName))['x']
        
        out_array      = np.zeros((patch_mat.shape[0],cue_img.shape[0],cue_img.shape[1]))
        #pdb.set_trace()
        
        if(ph < 0 and pw < 0):
            ph          = patch_mat.shape[1]
            pw          = patch_mat.shape[2]
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
                patch_mat      = (loadmat(patch_fileName))['x']
                patch_idx += 1
                if(len(out_array.shape) > 2):
                    out_array[:, h : h + ph, w : w + pw]  = patch_mat
                else:
                    out_array[h : h + ph, w : w + pw]     = patch_mat
                
                #print("file %s: idx:%d" % (fil,patch_idx))
                #pdb.set_trace()
                
            
            # Column boundary patches
            patch_fileName = os.path.join(in_datapath,fname+'_'+fmt.format("%04d" % patch_idx)+'.'+ext)
            patch_mat      = (loadmat(patch_fileName))['x']
            patch_idx += 1
            if(len(out_array.shape) > 2):
                out_array[:, h : h + ph, -pw:]    = patch_mat
            else:
                out_array[h : h + ph, -pw:]       = patch_mat
        
        # Row boundary patches
        for w in wrange:
            # print(h, w, out_array.shape[2], out_array.shape[3])
            patch_fileName = os.path.join(in_datapath,fname+'_'+fmt.format("%04d" % patch_idx)+'.'+ext)
            patch_mat      = (loadmat(patch_fileName))['x']
            patch_idx += 1
            
            if(len(out_array.shape) > 2):
                out_array[:, -ph:, w : w + pw]    = patch_mat
            else:
                out_array[-ph:, w : w + pw]       = patch_mat
        
        patch_fileName = os.path.join(in_datapath,fname+'_'+fmt.format("%04d" % patch_idx)+'.'+ext)
        patch_mat      = (loadmat(patch_fileName))['x']
        patch_idx += 1
        
        if(len(out_array.shape) > 2):
            out_array[:, -ph:, -pw:]  = patch_mat
        else:
            out_array[-ph:, -pw:]     = patch_mat
        
        out_mat         = {}
        out_mat['Dc']   = out_array
        out_fileName = os.path.join(out_datapath,fname+'.'+ ext)
        savemat(out_fileName,out_mat)
        
        print(" ====== Processing: %s ======" % (fname+'.'+ ext))
        #pdb.set_trace()

'''
Usage:
python dibcoImgJoin.py
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str, default='../datasets/xDIBCO/xOutput_pdfs')
    parser.add_argument("-o", "--outdir", type=str, default='../datasets/xDIBCO/jOutput_pdfs')
    parser.add_argument("-c", "--cuedir", type=str, default='../datasets/DIBCO/pngs/img')
    #parser.add_argument("-ph", type=int, default=200)
    #parser.add_argument("-pw", type=int, default=256)
    
    args = parser.parse_args()
    
    args.step   = 0.0
    args.ph     = -1
    args.pw     = -1
    
    getMatTiles(args)
    