# from PIL import Image
import numpy as np
import time

def to_grayscale(im):
    return im.convert("F")

def img_to_matrix(im):
    """Returns numpy matrix with dimensions of size of bands of image"""
    return np.matrix(im)

def flatten(lst):
    lst = [tuple(item) for sublist in lst for item in sublist]

def create_matrix_border(im_matrix, border_val = 0, top = 1, bottom = 1, left = 1, right = 1):
    row, col = im_matrix.shape
    horiz_arr = np.zeros((1,col))
    vert_arr = np.zeros((1,row))

    if top:
        im_matrix[0] = horiz_arr
    if bottom:
        im_matrix[row-1] = horiz_arr

    im_matrix = im_matrix.transpose()

    if left:
        im_matrix[0] = vert_arr
    if right:
        im_matrix[col-1] = vert_arr

    return im_matrix.T

def benchmark(fn):
    import time
    def inner(*args):
        start = time.clock()
        retval = fn(*args)
        end = time.clock()
        print("Time of function '" + fn.__name__ + "' is: " + str(end - start) + " s")
        return retval
    return inner
