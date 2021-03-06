from PIL import Image
import numpy as np
import utils
# import argparser

def main():
    im = Image.open(im_path)
    new_im = CARVE(im, (x,y))
    new_im.save(save_path, file_type = "JPEG")

def CARVE(im, res = (None,None)):
    """
    @im: Image to CARVE
    @res: 'Resolution' -> tuple of final dimensions of im
    """
    w, h = im.size
    out_w, out_h = res[0], res[1]

    columns_removed = 0
    rows_removed = 0


    r,g,b = b(im.split)
    r = np.matrix(r)
    g = np.matrix(g)
    b = np.matrix(b)
    c_matrix = np.array(list(im.getdata())).reshape(im.size)

    for i in range(1,w - out_w + 1):
        current_matrix_size = (h - rows_removed,w - columns_removed, 3)


        energy = cumilative_energy_matrix(r) + cumilative_energy_matrix(g) + cumilative_energy_matrix(b)
        min_path = find_seam(energy)
        if i % 5 == 0:
            mark_seam(c_matrix, min_path)
        c_matrix = remove_seam(c_matrix, min_path)

    im = Image.new("RGB", res)
    im.putdata(data)
    im.show()


    return im;


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



def find_seam(energy_matrix):
    """Expects first row to be all 0s"""
    h, w = energy_matrix.shape #(row,column) -> (height, width)
    path = []
    for r in range(1, h):
        for c in range(w):
            adj = [a  if (a >=0 and a < w) else c for a in range(c-1, c + 2)]

            energy_matrix[r,c] += min(energy_matrix[r-1,adj[0]], energy_matrix[r-1,adj[1]], energy_matrix[r-1,adj[2]])
    # Go Back Up creating the shortest path as you go
    col = np.argmin(energy_matrix[h - 1])
    path.append((h-1, col))

    for r in range(1, h): # Includes 0 not -1
        row = h - r - 1
        adj = [a  if (a >=0 and a < w) else c for a in range(col-1, col + 2)]
        col += np.argmin(energy_matrix[row, min(adj):max(adj) + 1]) - 1
        path.append((row, col)) #(r,c)

    path.sort()
    # print(path)
    return path



def mark_seam(matrix, path):
    for elem in path:
        matrix[elem[0], elem[1]] = [255,0,0]
    matrix.reshape(1,-1,3)
    data = [tuple(item) for sublist in matrix.tolist() for item in sublist]
    marked_im = Image.new("RGB", (matrix.shape[1], matrix.shape[0]))
    marked_im.putdata(data)
    marked_im.show()

def remove_seam(matrix, path):
    """
    @im: Image
    @dir: direction of seam deletion
    """
    # print("removing seam path: ",path)
    shape = matrix.shape
    matrix.reshape(1,-1,3)
    lst = [tuple(item) for sublist in matrix.tolist() for item in sublist]
    num_rem = 0
    for elem in path:
        w, h = shape[1], shape[0]
        row = elem[0]
        col = elem[1]
        lst.pop(row * w + col - num_rem)
        num_rem += 1
    return np.array(lst).reshape(shape)

# a = np.matrix([[1,2,3],[3,4,5],[5,6,7]])
# b = cumilative_energy_matrix(a)
# print(b)


im = Image.open("small1.jpg")
CARVE(im, (635, 640))
