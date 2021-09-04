import numpy as np

class vp_hash_table:
    def __init__(self, h, w):
        self.ht = np.zeros(shape=(h*w+w), dtype=tuple)
        self.h = h
        self.w = w
        for y in range(h):
            for x in range(w):
                self.ht[y*w+x] = (y,x)
    
    def vertice2pix(self, v):
        return self.ht[v]

    def pix2vertice(self, x, y):
        return y*self.w+x