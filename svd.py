import numpy as np


# ### 1.1. Orthogonal Matrices

def is_orthogonal(m1):
    """ Given a matrix, return True if the matrix is an orthogonal matrix, False otherwise."""
    
    is_orthogonal = None
    
    if np.linalg.det(m1) == 1 or np.linalg.det(m1) == -1:
        is_orthogonal = True
    else:
        is_orthogonal = False

    return is_orthogonal


# ### 1.2. Invertible Matrices 

def is_invertible(m1: np.ndarray):
    """Given any square n x n matrix, determine whether or not a matrix is invertible. Return True if the 
    matrix is invertible, False otherwise."""
    
    is_invertible = True
    
    return is_invertible if np.linalg.det(m1) != 0 else False


# ### 1.3 Powers of a Matrix 
def powers_of_matrix(s, a, s_inv, power):
    """Given an eigen-decomposition, recover the original matrix and return the specified power of the matrix. Return
    both answers (original matrix, power of the matrix) in tuple format."""
    
    powers_of_matrix = (None, None)
    
    
    eigenvalues, _ = np.linalg.eig(a)

    diag = np.diag(eigenvalues)
    diag_powered = diag**power
    
    matrix_powered = s @ diag_powered @ s_inv
    
    orig_matrix = s @ diag @ s_inv
    
    powers_of_matrix = orig_matrix, matrix_powered
    
    return powers_of_matrix


# ### 1.4 Powers of a Matrix, continued

def powers_of_matrix_eig(eigenvalues, eigenvectors, power):
    """Given both the eigenvectors and eigenvalues of a matrix, compute the eigenvectors and eigenvalues for the 
    same matrix at the specified power. Return as a tuple in the form of (eigenvectors, eigenvalues)."""
    
    powers_of_matrix_eig = (None, None)
    
    
#     S = eigenvectors
    diag = np.diag(eigenvalues) 
#     S_inv = np.linalg.inv(S)
    
    diag_powered = diag**power
    powered_eigenvalues, _ = np.linalg.eig(diag_powered)
#     orig_matrix_powered = S @ diag_powered @ S_inv
    
    powers_of_matrix_eig = eigenvectors, powered_eigenvalues
    
    return powers_of_matrix_eig

# ### 1.5 Symmetric Matrices
def is_symmetric(m1: np.ndarray):
    """ Given a square n x n matrix, return True if the matrix is symmetric and False otherwise."""
    
    is_symmetric = None
    
    
    is_symmetric = (m1 == m1.T).all()

    return is_symmetric

# ### 1.6 Positive Definite Matrices
def positive_definite(m1: np.ndarray):
    """Given a n x n square matrix, check: 
      That the matrix is symmetric; 
        1. If the matrix is symmetric, return "positive semi-definite" if the matrix is positive semi-definite; 
        2. If the matrix is symmetric, return "positive definite" if the matrix is positive definite;
        3. If the matrix is symmetric but neither positive semi-definite nor positive definite, then "neither"
        4. If the matrix is not symmetric, return "not symmetric"
    """
    
    positive_definite = None
    
    eigenvalues, _ = np.linalg.eig(m1)

    
    if is_symmetric(m1):
        if all(eigenvalues) > 0:
            positive_definite = "positive definite"
        elif all(eigenvalues) >= 0:
            positive_definite = "positive semi-definite"
        else:
            positive_definite = "neither"
            
    else:
        positive_definite = "not symmetric"
        
    
    return positive_definite
    

# ---
# 
# ## Part 2 Applications - Markov Chains 
# A matrix can 
# represent a set of probability transitions from one state to another. Let's start with the probability state of you, in front of a vending machine. The probability of buying a soda, a bag of chips, gum, or nothing at all is dependant on what you bought the previous time:
# 
# - if you bought soda previously, the probability of you buying soda (again) is .30; chips: .40, gum: .20, nothing: .10
# - if you bought chips previously, the probabilities are: a soda: .25, chips: .20, gum: .30, nothing: .25
# - if you bought gum previously: soda: .10, chips: .40, gum: .20, nothing: .30
# - if you bought nothing previously: soda: .25, chips: .25, gum: .25, nothing: .25
# 
# This can be represented as a probability matrix of:
# 

snacking = np.array([[.30, .40, .20, .10], [.25, .20, .30, .25], [.10, .40, .20, .30], [.25, .25, .25, .25]]).T
snacking


# Your previous state can be represented as a vector. Let's say you drank a soda previously, which can be represented as shown below.

previous_state = np.array([1, 0, 0, 0])


# ### 2.1 First Time at the Machine
def first_time(snacking):
    """ What are the probabilities of you buying each item the first time you approach the machine?"""
    
    probabilities = None
    
#     probabilities = np.array([.25, .25, .25, .25])
    previous_state = np.array([0, 0, 0, 1])

    probabilities = np.dot(snacking, previous_state)
    
    
    return probabilities

# ### 2.2 Second Time at the Machine
# 

def second_time(snacking):
    """ What are the probabilities that you buy soda, chips, gum, or nothing after two (2) rounds at the machine, having 
    initially bought nothing?"""
    
    probabilities = None
    
#     eigenvals, eigenvecs = np.linalg.eig(snacking)
    
#     S = eigenvecs
#     diag = np.diag(eigenvals)
#     S_inv = np.linalg.inv(S)
    
    
#     powered_mat = S @ diag**2 @ S_inv
    
#     probabilities = np.array([.25, .25, .25, .25])
#     print('eigenvals', eigenvals)
#     print()
#     print('diag', diag)
#     return probabilities**2

    print('snacking', snacking)

    previous_state = np.array([0, 0, 0, 1])

    probabilities = np.dot(snacking, np.dot(snacking, previous_state))


    return probabilities

snacking = np.array([[.30, .40, .20, .10], [.25, .20, .30, .25], [.10, .40, .20, .30], [.25, .25, .25, .25]]).T
previous_state = np.array([0, 0, 0, 1])

np.dot(snacking, np.dot(snacking, previous_state))
snacking[:, -1]

# ### 2.3 Tenth Time at the Machine

def tenth_time(snacking):
    """All this math homework is making you hungry. After 10 rounds of this, what item has the highest probability 
    of being purchased by you? Return your answer as a string of one of the listed items: soda, chips, gum, nothing.
    """
    
    choice = None
    
    
    initial_state = np.array([0, 0, 0, 1])
    items = ['soda', 'chips', 'gum', 'nothing']
    x_n = initial_state
    for i in range(10):
        x_n = np.dot(snacking, x_n)
    choice = sorted(tuple(zip(items, x_n)), key = lambda x: x[1])[::-1][0][0]
    return choice

tenth_time(snacking)
# np.dot(snacking, first_time(snacking))


# ### 2.4 Markov Chains - Generalization
# Now that we've done this a couple of times, we can certainly generalize. Given a probability transition matrix and a starting state, can one generalize what probabilities exist for certain events after n iterations?  

def markov_event(transition_matrix, starting_sample, n):
    """ Given a probability transition matrix and the starting position, what is the probabilities of each event 
    after n transitions?"""
    
    probability = None
    
    x_n = starting_sample
    for i in range(n):
        x_n = np.dot(transition_matrix, x_n)
    probability = x_n.round(3)
    return probability

# ## Part 3 Applications - Dimensionality Reduction
# 
# Images can be seen as a set of matrices, one for each channel of color (e.g., 3 channels for an RGB image, 1 for grayscale, etc). SVD can be used for dimensionality reduction and we will implement the image reduction. 

# We're going to be manipulating this image in particular: 
import time
import numpy as np

from PIL import Image, ImageOps

filename = 'assets/umsi.png'

Image.open(filename)

# ### 3.1 Manipulating Images

def open_image(filename = 'assets/umsi.png'):
    """Open image using Pillow, convert the image to grayscale, and resize it to 520 x 360. The image returned is a numpy array. 
    """

    
#     im = np.array(Image.open(filename).convert("L").resize((520, 360)))
    im = np.array(Image.open(filename).convert('L').resize((520, 360)))
    return im

open_image().shape


# ### 3.2 Singular Value Decomposition

def svd(m1: np.ndarray, full_matrices: bool = True):
    """Given a matrix, perform SVD on it and return all its components (u, sigma, v) as a tuple. Return either 
    full matrices (or not) based on the boolean full_matrices parameter being passed into the function."""
    
    
    U, Sigma, VT = np.linalg.svd(m1, full_matrices)
    
    if not full_matrices:
        VT = VT.T
    return U, Sigma, VT

start = time.time()

stu_u, stu_s, stu_v = svd(open_image(filename), full_matrices = False)

assert stu_u.shape == (360, 360), f"""Q.3.2 SVD - Your shape for u should be (360,360); your shape is 
{stu_u.shape}."""
assert stu_s.shape == (360,), f"""Q.3.2 SVD - Your shape for s should be (360,); your shape is 
{stu_s.shape}."""
assert stu_v.shape == (520, 360), f"""Q.3.2 SVD - Your shape for v should be (520,360); your shape is 
{stu_v.shape}. Did you forget to not return full matrices when the full_matrices parameter is set to False?"""

end = time.time()
print("Time elapsed:",end - start)

# ### 3.3 Truncated SVD
def truncate(m1, k: int):
    """Given a matrix, perform SVD (using full matrices), truncate to k, reconstruct the matrix using 
    truncated values, and then pass it back into Pillow to reconstruct the image. Return a tuple of the 
    (reconstructed matrix, image).
    
    """
    
    
    
    
    U, S, V_T = np.linalg.svd(m1, full_matrices=True)
    U_k = U[:,:k]
    
    S_k = np.diag(S[:k])

    
    V_T_k = V_T[:k,:]
    
    A_k = U_k @ S_k @ V_T_k

    image_array = A_k
    
    image_array = image_array.astype('uint8')

    
    return image_array, Image.fromarray(image_array)

#     return image_array


"""Test code and tme to complete it."""

start = time.time()

stu_ans, stu_img = truncate(open_image(filename), k=5)

assert stu_ans.shape == (360, 520), f"""Q3.3, Truncated SVD - Your reconstructed matrix is the wrong shape;
it should be (360, 520), not {stu_ans.shape}."""

end = time.time()
print("Time elapsed:",end - start)

# Let's see what happens to your images when k = 360, hmm? 

filename = 'assets/umsi.png'
k = 360

# uncomment to view
# truncate(open_image(filename), k=k)[1]


# 360 looks pretty impressive, don't you think? What about 200?  

filename = 'assets/umsi.png'
k = 200

# uncomment to view
# truncate(open_image(filename), k=k)[1]

# 200 still is pretty recognizable! What about 50?  

filename = 'assets/umsi.png'
k = 50

# uncomment to view
# truncate(open_image(filename), k=k)[1]

# Wow, 50 is still pretty recognizable, although definitely lossy. What about 10? 

filename = 'assets/umsi.png'
k = 10
# uncomment to view
# truncate(open_image(filename), k=k)[1]

# Yeah.... seems to be 50 is our limit. But 50 is incredibly impressive!


