import numpy as np

class Matrix:
    """Interface of a matrix.

    This class provides only the matrix size N and a method for swapping
    two items. The actual storage of the matrix in memory is provided by
    subclasses in testing code.
    """


    def __init__(self, N):
        self.N = N
        self.MIN_N = 1

    def swap(self, i1, j1, i2, j2):
        """Swap elements (i1,j1) and (i2,j2)."""

        # Overridden in subclasses
        raise NotImplementedError

    def simple_quadrant_transpose(self, begin_h, end_h, begin_w, end_w):
        """Swap elements in square quadrant """
        for i in range(begin_h, end_h):
            for j in range(begin_h, i):
                self.swap(i, j, j, i)

    def transpose_quadrant(self, begin_h, end_h, begin_w, end_w):
        n_w = end_w - begin_w
        n_h = end_h - begin_h
        if n_w <= self.MIN_N or n_h <= self.MIN_N:
            self.simple_quadrant_transpose(begin_h, end_h, begin_w, end_w)
            return
        half_w = begin_w + int(np.floor(n_w / 2))
        half_h = begin_h + int(np.floor(n_h / 2))

        self.transpose_quadrant(begin_h, half_h, begin_w, half_w)
        self.transpose_quadrant(half_h, end_h, half_w, end_w)
        self.transpose_swap_quadrants(begin_h, half_h, half_w, end_w)

    def simple_quadrants_swap(self, begin_h, end_h, begin_w, end_w):
        """Swap elements in top right quadrant with bottom left quadrant"""
        for i in range(begin_h, end_h):
            for j in range(begin_w, end_w):
                self.swap(i, j, j, i)

    def transpose_swap_quadrants(self, begin_h, end_h, begin_w, end_w):
        """Given the upper right quadrant coordinates, swap it with lower left quadrant"""
        n_w = end_w - begin_w
        n_h = end_h - begin_h
        if n_w <= self.MIN_N or n_h <= self.MIN_N:
            self.simple_quadrants_swap(begin_h, end_h, begin_w, end_w)
            return
        half_w = begin_w + int(np.floor(n_w / 2))
        half_h = begin_h + int(np.floor(n_h / 2))
        
        self.transpose_swap_quadrants(begin_h, half_h, begin_w, half_w) # upper left  subquadrant of upper right quadrant
        self.transpose_swap_quadrants(begin_h, half_h, half_w, end_w)   # upper right ----------------||-----------------
        self.transpose_swap_quadrants(half_h, end_h, begin_w, half_w)   # lower left  ----------------||-----------------  
        self.transpose_swap_quadrants(half_h, end_h, half_w, end_w)     # lower right ----------------||-----------------

    def transpose(self):
        """Transpose the matrix."""
        self.transpose_quadrant(0, self.N, 0, self.N)



        
