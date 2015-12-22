from PIL import Image
import numpy as np
import time
import utils
color_im = Image.open("small1.jpg")
r,g,b = color_im.split()
r = np.matrix(r)
g = np.matrix(g)
b = np.matrix(b)
def cumilative_energy_matrix(init_matrix, previous = None, removed_seam = None):
    """
    >>> x = 5
    >>> x
    5
    """
    assert isinstance(init_matrix, np.matrix) , "Function takes a NUMPY matrix."

    def adj_sum(matrix):
        """ REQUIRES: numpy
            Sums ...
            Circular sum adj -> edges are summed with the element on the other side of the matrix
        """
        new_m = np.zeros(matrix.shape)

        new_m += np.roll(matrix,1) + np.roll(matrix,-1)#add right and left
        new_m += np.roll(matrix,1, 0) + np.roll(matrix,-1, 0) # add top and bottom

        matrix = np.roll(matrix,1,0)
        new_m += np.roll(matrix,1) + np.roll(matrix,-1)#add right and left
        matrix = np.roll(matrix,-2,0)
        new_m += np.roll(matrix,1) + np.roll(matrix,-1)#add right and left
        return new_m
    def grad_difference(matrix, axis = 0):
        """ Axis = 1 if x grad else 0"""
        new_m = np.zeros(matrix.shape)
        new_m += np.roll(matrix,1, axis) - np.roll(matrix,-1, axis)
        return new_m

    p = np.power
    dual_gradient = p(grad_difference(init_matrix, 0), 2) + p(grad_difference(init_matrix, 1), 2)
    return utils.create_matrix_border(dual_gradient, 0, 1, 0, 0, 0 )

cr = cumilative_energy_matrix(r)
cg = cumilative_energy_matrix(g)
cb = cumilative_energy_matrix(b)
energy = cr + cg + cb
r1 = [1,2,3,4]
r2 = [2,3,4,5]
r3 = [0,1,2,3]
s = time.clock()
print(energy[1])
stack = np.dstack((energy[1], np.roll(energy[1], 1), np.roll(energy[1],-1)))
# print(energy[4])
# print(stack[4])
print(stack)
print([min(i) for i in stack[0]])
print(time.clock() - s)
# print(min(stack[0,0]))
