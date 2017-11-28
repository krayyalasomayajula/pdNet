from cv2 import imwrite
__all__ = ['denseDeformationFieldFromSparse']

import numpy as np
import sys

small_ = 1e-100

Ufunction = [None, (lambda x: np.abs(x) ** 3), (lambda x: x ** 2 * np.where(x < small_, 0, np.log(x ** 2))), (lambda x: np.abs(x))]

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

from numpy.testing import *
import pdb
import cv2

def test_1DCreation():
    size = (10,)
    points = np.array(((2,), (8,)))
    displacements = [0, (-4,)]
    
    field = denseDeformationFieldFromSparse(size, points, displacements)

    for point, displacement in zip(points, displacements):
        assert_almost_equal(field[tuple(point.tolist())], displacement)
    
    print('Deformation Field is:' , field)
    
def test_2DCreation():
    size = (5, 5)
    points = np.array(((1., 1.), (4., 1.), (4., 4.), (1., 4.)))
    displacements = [(3., 0.), (0, 3) , (-3, 0), (0, -3)]
    
    field = denseDeformationFieldFromSparse(size, points, displacements)
    for point, displacement in zip(points, displacements):
        assert_almost_equal(field[tuple(point.tolist())], displacement)
    
    np.set_printoptions(precision=1,suppress=True,threshold=100000)
    print('Deformation Field is:', field)

def computeDisplacement(iPos, fPos):
    assert(iPos.shape[0] == fPos.shape[0])
    assert(iPos.shape[1] == fPos.shape[1])
    
    displacement = (iPos - fPos).tolist()
    
    return displacement
    
def test_Kanji():
    src = cv2.imread("kanji.png")
    if(src is None):
        sys.exit()
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    size = src.shape

    initPos = np.array([[197,41], 
                        [481,41],
                        [431,0],
                        [274,145],
                        [327,235],
                        [535,185],
                        [56,271],
                        [4,497],
                        [794,500]])
    
    finalPos = np.array([[190,63], 
                        [490,21],
                        [406,15],
                        [303,106],
                        [297,208],
                        [530,213],
                        [89,216],
                        [174,497],
                        [590,490]])
    
    points = initPos
    displacements = computeDisplacement(initPos, finalPos)
    
    #pdb.set_trace()
    
    field = denseDeformationFieldFromSparse(size, points, displacements)
    #for point, displacement in zip(points, displacements):
    #    assert_almost_equal(field[tuple(point.tolist())], displacement)
    
    np.set_printoptions(precision=1,suppress=True,threshold=100000)
    print('Deformation Field is:', field)
    
    #field = field.astype(np.int)
    
    absCords = np.zeros(field.shape)
    
    #pdb.set_trace()
    
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            absCords[i,j,:] = np.array([i,j])
    
    #pdb.set_trace()
    
    opImg = np.zeros(size, np.uint8)
    absCords = (absCords+field).astype(np.float32)
    
    dMap1, dMap2 = cv2.convertMaps(absCords[:,:,0], absCords[:,:,1],cv2.CV_32FC1)
    opImg = cv2.remap(src.T,dMap1,dMap2, cv2.INTER_LINEAR)
    
    cv2,imwrite('defo_Kanji.png',opImg)            
    
def test_2DCreation_bis():
    size = (10, 10)
    points = np.array(((5., 5.), (2., 8.), (8., 2.)))
    displacements = [0, (5, -4), (0, 2)]
    
    field = denseDeformationFieldFromSparse(size, points, displacements)
    for point, displacement in zip(points, displacements):
        assert_almost_equal(field[tuple(point.tolist())], displacement)
    
    print('Deformation Field is:', field)
    
def test_3DCreation():
    size = (10, 10, 10)
    points = np.array(((5., 5., 5.), (2., 8., 2.), (8., 2., 8.), (2., 2., 8.)))
    displacements = [0, (5, -4, 3), (0, 2, 0), -2]
    
    field = denseDeformationFieldFromSparse(size, points, displacements)
    
    for point, displacement in zip(points, displacements):
        assert_almost_equal(field[tuple(point.tolist())], displacement)
    
    print('Deformation Field is:', field)
    
if __name__ == "__main__":
    #test_1DCreation()
    #test_2DCreation()
    #test_3DCreation()
    test_Kanji()
