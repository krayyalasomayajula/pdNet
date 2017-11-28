import sys
import argparse
import math

import cv2
import pdb

import os
from os import listdir
from os.path import isfile, isdir, join
import random
import numpy as np

small_ = 1e-100

Ufunction       = [None, (lambda x: np.abs(x) ** 3), (lambda x: x ** 2 * np.where(x < small_, 0, np.log(x ** 2))), (lambda x: np.abs(x))]

MIN_MAX = True

if(MIN_MAX):
    scaleFactors    = [('0p4',0.4),
                       ('0p5',0.5), 
                       ('0p65',0.65),
                       ('0p8',0.8),
                       ('1p0',1.0),
                       ('1p1',1.1),
                       ('1p25',1.25),
                       ('1p5',1.5)]
else:
    scaleFactors    = [('0p5',0.5),
                       ('0p8',0.8)]
    
def dist2hd(x, y):
    """
    Generate a 'coordinate' of the solution at a time
    """
    d = np.zeros((x.shape[0], y.shape[0]), dtype=x.dtype)
    for i in xrange(x.shape[1]):
        diff2 = x[:, i, None] - y[:, i]
        diff2 **= 2
        d += diff2
    np.sqrt(d, d)
    return d

def denseDeformationFieldFromSparse(size, points, displacements):
    """
    Creates a smooth deformation field based on :
    - its size
    - the points where the deformation is known
    - the displacement at those points
    size must be of the same size than the number of coordinates of points.
    This function uses Thin Plates to interpolate the field (cf Bookstein 1989 doi:10.1109/34.24792)
    """
    # Computation of the coefficients that will be stored in function
    points = np.asfarray(points)
    distances = dist2hd(points, points)
    totalsize = sum(points.shape)
    L = np.zeros((totalsize + 1, totalsize + 1))
    L[:len(points), :len(points)] = Ufunction[len(size)](distances)
    L[len(points), :len(points)] = 1
    L[:len(points), len(points)] = 1
    L[len(points) + 1:, :len(points)] = points.T
    L[:len(points), len(points) + 1:] = points
    Y = np.zeros((points.shape[0] + len(size) + 1, points.shape[1]))
    for i in range(0, len(displacements)):
        Y[i] = displacements[i]
    function = np.linalg.solve(L, Y)
    
    # Creation of the dense deformation field
    field = np.empty(size + (len(size),))
    for indice in range(len(size)):
        coeffs = np.arange(0, size[indice], dtype=int)
        newaxises = [1] * (len(size) - indice - 1)
        coeffs.shape = [coeffs.shape[0]] + newaxises
        field[..., indice] = coeffs
        
    view = field.reshape(-1, len(size))
    distances = dist2hd(view, points)
    distances = Ufunction[len(size)](distances)
    view[:] = function[len(points)] + np.dot(distances, function[:len(points)]) + np.dot(view, function[len(points) + 1:])
    
    return field

def computeDisplacement(iPos, fPos):
    assert(iPos.shape[0] == fPos.shape[0])
    assert(iPos.shape[1] == fPos.shape[1])
    
    displacement = (iPos - fPos).tolist()
    
    return displacement
    
'''
     ____________________ ____________________                     ____________________ 
    | Q11 __ __ __ __    | Q12                |                   | Q1N                |
    |    |     |     |   |                    |                   |                    |
    |    | q1  | q2  |   |                    |                   |                    |
    |     -- -- -- --    |                    |      ......       |                    |
    |    | q3  | q4  |   |                    |                   |                    |
    |    |__ __|__ __|   |                    |                   |                    |
    |____________________|____________________|                   |____________________|
    | Q21                |Q22                 |                   | Q2N                |
    |                    |                    |                   |                    |
    |                    |                    |                   |                    |
    |                    |                    |      ......       |                    |
    |                    |                    |                   |                    |
    |                    |                    |                   |                    |
    |____________________|____________________|                   |____________________|
          ..     :                   :                                        :
          ..     :                   :               ......                   :
          ..     :                   :                                        :
     ____________________ ____________________                     ____________________ 
    | QM1                |QM2                 |                   | QMN                |
    |                    |                    |                   |                    |
    |                    |                    |                   |                    |
    |                    |                    |      ......       |                    |
    |                    |                    |                   |                    |
    |                    |                    |                   |                    |
    |____________________|____________________|                   |____________________|
                                                                                        
'''
def quadrantDeformation(xscale,xfil,ximg_gt):
    img1 = cv2.imread(join(ximg_gt[0],xfil))
    img2 = cv2.imread(join(ximg_gt[1],xfil))
    
    #pdb.set_trace()
    assert(img1.shape[0] == img2.shape[0]), 'quadrantDeformation[assert1]: Size mismatch along dim1'
    assert(img1.shape[1] == img2.shape[1]), 'quadrantDeformation[assert2]: Size mismatch along dim2'
    
    ximg1   = cv2.resize(img1, None, fx=xscale, fy=xscale, interpolation = cv2.INTER_CUBIC)
    ximg2   = cv2.resize(img2, None, fx=xscale, fy=xscale, interpolation = cv2.INTER_CUBIC)
    
    ximg_shape   = (ximg1.shape[0],ximg1.shape[1])
    
    EXTN_BORDER = 200
    ximg1 = cv2.copyMakeBorder(ximg1,EXTN_BORDER,EXTN_BORDER,EXTN_BORDER,EXTN_BORDER,cv2.BORDER_REPLICATE)
    ximg2 = cv2.copyMakeBorder(ximg2,EXTN_BORDER,EXTN_BORDER,EXTN_BORDER,EXTN_BORDER,cv2.BORDER_REPLICATE)
    
    img_shape   = (ximg1.shape[0],ximg1.shape[1])
    
    #print('The shape is ', img_shape)
    BORDER_FACTOR = 0.3 # should be less than 0.5 (2 borders)  
    border      = {'x':int(200*xscale*BORDER_FACTOR) ,'y': int(100*xscale*BORDER_FACTOR)}
    patch_shape = (200*xscale, 100*xscale)
    
    def getQuadrantsList(ximg_shape, xpatch_shape,xborder):
        ph = int(xpatch_shape[0])
        pw = int(xpatch_shape[1])
        
        h_strides = int(np.ceil(ximg_shape[0]/float(ph)) - 1)
        w_strides = int(np.ceil(ximg_shape[1]/float(pw)) - 1)
                    
        hrange = range(0, h_strides*ph, ph)
        wrange = range(0, w_strides*pw, pw)
        
        n_patches_h = len(hrange)
        n_patches_w = len(wrange)
        n_patches = n_patches_h * n_patches_w
        
        Qij    = {'x0': 0,
                  'x1': 0,
                  'y0': 0,
                  'y1': 0
                  }
        quadrants = [Qij]*n_patches
        
        patch_idx = 0
        for h in hrange:
            for w in wrange:
                quadrants[patch_idx] =  {'x0': h+xborder['x'],
                                         'x1': h + ph-xborder['x'],
                                         'y0': w+xborder['y'],
                                         'y1': w + pw -xborder['y']
                                        }
                patch_idx += 1
        
        return quadrants
        
    quadrants = getQuadrantsList(img_shape, patch_shape,border) # list of quadrants to iterate
    movements = ['move_q1', 'move_q2', 'move_q3', 'move_q4'] # list of movements to iterate
    
    #pdb.set_trace()
    
    '''
        The idea is to create 4 deformation fields that are formed due to
        moving:
                a random point in q1 -> a random point in q2/q3/q4
                a random point in q2 -> a random point in q1/q3/q4
                a random point in q3 -> a random point in q1/q2/q4
                a random point in q4 -> a random point in q1/q2/q3
        in each of Q1,Q2,Q3,Q4 respectively 
    '''
    def pickMove(xquads,xmove):
        n_moves = len(xquads)-1
        if(xmove == 'move_q1'):
            del_idx = 0
        if(xmove == 'move_q2'):
            del_idx = 1
        
        if(xmove == 'move_q3'):
            del_idx = 2
        
        if(xmove == 'move_q4'):
            del_idx = 3
            
        curr_xy = xquads[del_idx]
        del xquads[del_idx]
        rnd_idx = random.randint(0,n_moves-1) # -1 to accomodate python indexing
        next_xy = xquads[rnd_idx]
        
        return(curr_xy, next_xy)
    
    deformed_op = {'img':[],
                   'gt':[]
                   }
    
    for move in movements:
        init_Pos    = np.zeros([len(quadrants),2])
        final_Pos   = np.zeros([len(quadrants),2])
        
        cord_idx = 0
        for quad in quadrants:
            q1 = {'x': random.randint(quad['x0'],math.floor((quad['x1']+quad['x0'])/2.0)),
                  'y': random.randint(quad['y0'],math.floor((quad['y1']+quad['y0'])/2.0))}
             
            q2 = {'x': random.randint(quad['x0'],math.floor((quad['x1']+quad['x0'])/2.0)),
                  'y': random.randint(math.floor((quad['y1']+quad['y0'])/2.0), quad['y1'])}
            
            q3 = {'x': random.randint(math.floor((quad['x1']+quad['x0'])/2.0),quad['x1']),
                  'y': random.randint(quad['y0'],math.floor((quad['y1']+quad['y0'])/2.0))}
             
            q4 = {'x': random.randint(math.floor((quad['x1']+quad['x0'])/2.0),quad['x1']),
                  'y': random.randint(math.floor((quad['y1']+quad['y0'])/2.0), quad['y1'])}
            
            q_list = [q1,q2,q3,q4]
            
            #pdb.set_trace()
            curr_quad, next_quad = pickMove(q_list,move)
            
            init_Pos[cord_idx,0]    = curr_quad['x']
            init_Pos[cord_idx,1]    = curr_quad['y']
            final_Pos[cord_idx,0]   = next_quad['x']
            final_Pos[cord_idx,1]   = next_quad['y']
            
            cord_idx += 1    
    
        points = init_Pos
        displacements = computeDisplacement(init_Pos, final_Pos)
        
        #pdb.set_trace()
        
        field = denseDeformationFieldFromSparse(img_shape, points, displacements)
        
        '''
        np.set_printoptions(precision=1,suppress=True,threshold=100000)
        print('Deformation Field is:', field)
        '''
        absCords = np.zeros(field.shape)
        
        #pdb.set_trace()
        
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                absCords[i,j,:] = np.array([i,j])
        
        #pdb.set_trace()
        
        absCords = (absCords+field).astype(np.float32)
        dMap1, dMap2 = cv2.convertMaps(absCords[:,:,0], absCords[:,:,1],cv2.CV_32FC1)
        
        # Initialize output buffers
        opImg1 = 0.0 * ximg1
        opImg2 = 0.0 * ximg2
        
        if(len(opImg1.shape)>2):
            for i in range(opImg1.shape[2]):
                opImg1[:,:,i] = cv2.remap((ximg1[:,:,i]).T,dMap1,dMap2, cv2.INTER_LINEAR)
                opImg2[:,:,i] = cv2.remap((ximg2[:,:,i]).T,dMap1,dMap2, cv2.INTER_LINEAR)
            
            opImg2 = cv2.cvtColor(opImg2.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            ret,th_otsu = cv2.threshold(opImg2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            opImg1 = opImg1[EXTN_BORDER:EXTN_BORDER+ximg_shape[0],EXTN_BORDER:EXTN_BORDER+ximg_shape[1],:]
            opImg2 = th_otsu[EXTN_BORDER:EXTN_BORDER+ximg_shape[0],EXTN_BORDER:EXTN_BORDER+ximg_shape[1]]
        else:
            opImg1 = cv2.remap((ximg1).T,dMap1,dMap2, cv2.INTER_LINEAR)
            opImg2 = cv2.remap((ximg2).T,dMap1,dMap2, cv2.INTER_LINEAR)
            
            ret,th_otsu = cv2.threshold(opImg2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            opImg1 = opImg1[EXTN_BORDER:EXTN_BORDER+ximg_shape[0],EXTN_BORDER:EXTN_BORDER+ximg_shape[1]]
            opImg2 = th_otsu[EXTN_BORDER:EXTN_BORDER+ximg_shape[0],EXTN_BORDER:EXTN_BORDER+ximg_shape[1]]
        
        deformed_op['img'].append(opImg1)
        deformed_op['gt'].append(opImg2)
        
    return deformed_op

def augmentImgs(x_args):
    in_path         = os.path.abspath(x_args.indir)
    out_path        = os.path.abspath(x_args.outdir)
    
    out_img_path    = join(out_path, 'img')
    out_gt_path     = join(out_path, 'gt')
    
    try:
        os.stat(out_path)
    except:
        os.makedirs(out_path)
    
    try:
        os.stat(out_img_path)
    except:
        os.makedirs(out_img_path)
    
    try:
        os.stat(out_gt_path)
    except:
        os.makedirs(out_gt_path)
    
    #pdb.set_trace()
    
    img_gt      = [join(in_path,dir) for dir in listdir(in_path) if isdir(join(in_path, dir))]
    fil_list    = sorted(listdir(img_gt[0]))
    
    for fil in fil_list:
        print('<augmentImgs> Processing file: %s' % fil)
        for sname,scale in scaleFactors:
            aug_op = quadrantDeformation(scale,fil,img_gt)
            
            #pdb.set_trace()
            n_deformations = 0
            for i in range(len(aug_op['img'])):
                fname       = fil.split('.')
                op_fname   = fname[0]+'_'+sname+'_'+'df'+str(n_deformations)+'.'+fname[1]
                
                op_img_fname = join(out_img_path,op_fname)
                op_gt_fname  = join(out_gt_path,op_fname)
                
                cv2.imwrite(op_img_fname,aug_op['img'][i])
                cv2.imwrite(op_gt_fname,aug_op['gt'][i])
                
                n_deformations +=1
'''
Usage:
python bickleyAugment.py -i ../datasets/DIBCO/pngs/train -o ../datasets/DIBCO/augmented/train
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str, default=[''])
    parser.add_argument("-o", "--outdir", type=str, default=[''])
    
    args = parser.parse_args()
    
    augmentImgs(args)
    
