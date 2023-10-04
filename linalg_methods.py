from numpyl import numpy as np
import math

# ### Dot Product

def dot(v1: np.ndarray, v2: np.ndarray):
    """Calculate and return the dot product of two vectors."""
    
    dot_product = None
    
    # 
    dot_product = np.dot(v1, v2)
    
    return dot_product

def l2_norm(v1: np.ndarray):
    """Calculate the l2 (Euclidean) norm. """
    
    norm = None
    
    # 
    norm = math.sqrt(sum(np.square(v1)))
    
    return norm

def l1_norm(v1: np.ndarray):
    """Calculate the l1 (Manhattan) norm for a given vector."""
    
    norm = None
    
    # 
    norm = sum(abs(v1))
    
    return norm

# ### Normalization

def normalize(v1: np.ndarray):
    """Normalize a non-zero 1d vector. Length is defined here as its l2 norm."""
    
    norm_vector = None
    
    # 
    norm_vector = v1 / l2_norm(v1)
    
    return norm_vector

# ### 1.4 Orthogonal Projection

def orth_projection(a1: np.ndarray, a2: np.ndarray):
    "Calculate the orthogonal projection of column vector a2 onto line spanned by column vector a1."
    
    proj = None
    
    # 
    orig_shape = a1.shape
#     proj = ((dot(a2.T, a1)) / (l2_norm(v1)**2)) * a1
    a1 = np.array(a1).flatten()
    a2 = np.array(a2).flatten()
    proj = (dot(a2, a1) / dot(a1, a1)) * a1
    return proj.reshape(orig_shape)

v1 = np.array([[2, -2, -1]]).T
v2 = np.array([[4, 2, 0]]).T
stu_ans = orth_projection(v1, v2)

assert stu_ans.shape==(3, 1), f"Your matrix shape should be (3, 1), not {stu_ans.shape}."
assert np.all(stu_ans==np.array([[8/9, -8/9, -4/9]]).T), """Orthogonal Projection Test - 
Response is incorrect."""

# ### Gram Schmidt 

# can use previous functions (normalize & orthogonal projections) to solve 
# 
# 
# solve for u1/u2/u3 and stack vectors at the end
# 
# loop through 
#  - use orthogonal projection
#  - use normalization


def gram_schmidt(m1: np.ndarray):
    """Implement Gram-Schmidt where the set of vectors are stored as column vectors in m1. You can assume that no vector 
    will be the zero vector for this problem."""
    
    gram_schmidt = None
    
    # 
    v0 = m1[:, 0]
    u0 = (1 / l2_norm(v0))*v0
    orthonormal_basis_vecs = [u0]
    for i in range(1, len(m1)):
        vi = m1[:,i]
        if i == 1:
            yi = vi - orth_projection(m1[:,i-1], vi)
            ui = (1 / l2_norm(yi))*yi
            orthonormal_basis_vecs.append(ui)
        else:
            curr_orthos = [sum(orth_projection(orthonormal_basis_vecs[j], vi)) for j in range(i)]
            sum_of_curr_orthos = [0] * vi.shape[0]
            for j in range(i):
                sum_of_curr_orthos = np.add(sum_of_curr_orthos, (orth_projection(orthonormal_basis_vecs[j], vi)))
            yi = vi - sum_of_curr_orthos
            ui = (1 / l2_norm(yi))*yi
            orthonormal_basis_vecs.append(ui)  
    gram_schmidt = np.array(orthonormal_basis_vecs).T
    return gram_schmidt

m1 = np.array([[2, -2, -1], [4, 2, 0], [3, 0, 4]], dtype=np.float64).T

stu_ans = gram_schmidt(m1)
assert stu_ans.shape==(3, 3), f"Your matrix shape should be (3, 3), not {stu_ans.shape}."
assert np.all(np.isclose(stu_ans, np.array([[0.6666666666666666, -0.6666666666666666, -0.3333333333333333], 
                 [0.728810888813495, 0.6767529681839596, 0.1041158412590707], 
                 [0.15617376188860604, -0.31234752377721214, 0.9370425713316364]]).T)), """Gram Schmidt 
                 Test - Response is incorrect."""


# ## Word2Vec 

import pickle

with open('assets/madwords', 'rb') as f:
    words = pickle.load(f)

# uncomment to view - yes, that's right, the vector for pickle for the words we just unpickled: 

# print(words['pickle'])

# words.keys()
len(words['consolidating'])

# words.get('consolidating')

# len(words.values())

# for w in words:
#     print(words[w])

# words['prince']


# ### Cosine Similarity

def cosine_similarity(v1: np.ndarray, v2: np.ndarray):
    """Calculate the cosine similarity between two vectors.  Scipy is also not made available for you here as well."""
    
    cosine_similarity = None
    
    # 
    cosine_similarity = dot(((v1 / l2_norm(v1)).T), (v2 / l2_norm(v2)))
    return cosine_similarity

v1 = np.array([3, 4])
v2 = np.array([5, 12])

stu_ans = cosine_similarity(v1, v2)

assert np.isclose(stu_ans, 0.9692307692307692), """Q2.1, Cosine Similarity 
Test - Response is incorrect."""

# cosine_similarity(words['emperor'], words.keys())
cosine_similarity(words['emperor'], words['Emperor'])


# ### Most Similar Words

def calc_cos_sim_between_input_word_and_model_words(vec_to_compare, model, most_similar_words):
    for word in model:
        cos_sim = cosine_similarity(vec_to_compare, model[word])
        most_similar_words.append((word, cos_sim))
    return most_similar_words
        
def most_similar(list_of_word_vectors, model = words):
    """Calculate the cosine similarity between a list of words with the loaded word vectors and return the top 10 most 
    similar words ranked by highest similarity in descending order. When multiple words are passed in the 
    list_of_word_vectors, the vector used for the cosine similarity calculation should be the mean of the vectors. Do 
    not remove matching words in the most similar list."""
    
    most_similar_words = []
    
    # 
    if len(list_of_word_vectors) > 1:
        curr_vec = \
        np.mean([np.array(words[word]) for word in list_of_word_vectors], axis=0)
        most_similar_words = calc_cos_sim_between_input_word_and_model_words(curr_vec, model, most_similar_words)
    else:
        curr_vec = model[list_of_word_vectors[0]]
        most_similar_words = calc_cos_sim_between_input_word_and_model_words(curr_vec, model, most_similar_words)
        
    most_similar_words = [tup[0] for tup in sorted(most_similar_words, key=lambda x: x[1], reverse=True)[:10]]

    return tuple(most_similar_words)
    


with open('assets/madwords', 'rb') as f:
    words = pickle.load(f)
    
list_of_words = ['emperor']
stu_ans = most_similar(list_of_words, words)

assert (stu_ans[0]=='emperor'), """Your first word in this list should be 'emperor.' After all, a word is most similar
to itself!"""
assert (stu_ans==('emperor', 'Emperor', 'imperial', 'monarch', 'prince', 
                  'deity', 'monk', 'princes', 'tyrant', 'ruler')), """Q2.2, Most Similar - Your answer is incorrect."""

# What happens when you pass in a word that _might_ have different meanings in multiple contexts? Well, let's see:

list_of_words = ['haircut']
most_similar(list_of_words, words)


# Seems to mostly make sense, right? 'haircut' by itself would largely center around the act of cutting hair. Now what happens when we change the context of 'haircut' when we add another word?

list_of_words = ['haircut', 'interbank']
most_similar(list_of_words, words)


# Now most closely related words change quite a bit! 'haircut' in the context of the financial markets (particularly related to short-term lending, e.g., the repo and reverse repo markets) is now instead related to the discount on an asset from its market value for a repo rather than the act of cutting hair.



