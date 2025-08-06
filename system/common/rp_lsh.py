"""
Random Projection-based LSH. Implementation found online at https://github.com/gamboviol/lsh.
"""

import numpy as np
import random
from scipy.spatial import distance

class CosineHash:

    def __init__(self,r):
        self.r = r

    def hash(self,vec):
        return self.sgn(np.dot(vec,self.r))

    def sgn(self,x):
        return int(x>0)


class CosineHashFamily:
    """
    from https://github.com/gamboviol/lsh/blob/master/lsh.py
    based on method described by Charikar at https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf
    """

    def __init__(self,d):
        self.d = d

    def create_hash_func(self):
        # each CosineHash is initialised with a random projection vector
        return CosineHash(self.rand_vec())

    def rand_vec(self):
        return [random.gauss(0,1) for i in range(self.d)]

    def combine(self,hashes):
        """ combine by treating as a bitvector """
        return sum(2**i if h > 0 else 0 for i,h in enumerate(hashes))

    def concat(self, hashes):
        bitstr = ""
        for i in hashes:
            bitstr += str(i)
        return bitstr

def hamming(s1, s2):
    """
    https://stackoverflow.com/questions/31007054/hamming-distance-between-two-binary-strings-not-working
    Calculate the Hamming distance between two bit strings
    """
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def hash_point(fam, hash_funcs, p):
    return fam.concat([h.hash(p) for h in hash_funcs])


def get_lsh_hash_dist(v1, v2, k):
    """
    Assume v1 and v2 have same dimensionality

    Returns Hamming distance between arccos hashes of v1 and v2
    """
    d = v1.shape[0]
    fam = CosineHashFamily(d)
    hash_funcs = [fam.create_hash_func() for h in range(k)]
    h1 = hash_point(fam, hash_funcs, v1)
    h2 = hash_point(fam, hash_funcs, v2)
    d = hamming(h1, h2)
    return d

# p1 = np.random.randint(0, 10000, (1, 512)).reshape(-1)
# p2 = np.random.randint(0, 10000, (1, 512)).reshape(-1)

# # cossim = -1*(distance.cosine(p1, p2) - 1)
# print(cossim)
# print(res1, res2)
# print(hamming(res1, res2))





