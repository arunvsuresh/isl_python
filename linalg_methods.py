from numpyl import numpy as np

import math


# ### Dot Product of 2 vectors

def dot(v1: np.ndarray, v2: np.ndarray):
    """Calculate and return the dot product of two vectors."""
    
    dot_product = None
    
    dot_product = np.dot(v1, v2)
    
    return dot_product


# ### Norms of vectors
def l2_norm(v1: np.ndarray):
    """Calculate the l2 (Euclidean) norm. """
    
    norm = None
    
    norm = math.sqrt(sum(np.square(v1)))
    
    return norm


def l1_norm(v1: np.ndarray):
    """Calculate the l1 (Manhattan) norm for a given vector."""
    
    norm = None
    
    norm = sum(abs(v1))
    
    return norm


# ### Normalization of vectors

def normalize(v1: np.ndarray):
    """Normalize a non-zero 1d vector. Length is defined here as its l2 norm."""
    
    norm_vector = None
    
    norm_vector = v1 / l2_norm(v1)
    
    return norm_vector

# ### Orthogonal Projection of vectors 
def orth_projection(a1: np.ndarray, a2: np.ndarray):
    "Calculate the orthogonal projection of column vector a2 onto line spanned by column vector a1."
    
    proj = None
    
    proj = ((dot(a2.T, a1)) / (l2_norm(v1)**2)) * a1
    
    return proj

v1 = np.array([[2, -2, -1]]).T
v2 = np.array([[4, 2, 0]]).T
stu_ans = orth_projection(v1, v2)

assert stu_ans.shape==(3, 1), f"Your matrix shape should be (3, 1), not {stu_ans.shape}."
assert np.all(stu_ans==np.array([[8/9, -8/9, -4/9]]).T), 

del stu_ans
