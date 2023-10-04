#!/usr/bin/env python
# coding: utf-8

# ## Part 1: Vector Norms and Orthogonal Vectors Review (55 points)
# 
# For assignment 1, some parts of numpy (specifically, np.linalg in its entirety) are not available to you in order to test you on the material below and its import . However, for ease of use regarding certain data structures (e.g., transpose), most parts of numpy outside the linalg library are available to you. As some of these problems certianly build on previous questions, feel free to reuse the methods you implemented in later questions or reimplement them in its entirety in later questions. 

# In[1]:


from numpyl import numpy as np


# In[2]:


# This cell is to import numpy-light for the autograder, not for grading


# Here, we can start with an easy warmup...

# ### 1.1 Dot Product (3 points)

# In[3]:


def dot(v1: np.ndarray, v2: np.ndarray):
    """Calculate and return the dot product of two vectors."""
    
    dot_product = None
    
    # YOUR CODE HERE
    dot_product = np.dot(v1, v2)
    
    return dot_product


# In[4]:


# Autograder test


# ### 1.2 Norms (10 points)

# In[5]:


import math
def l2_norm(v1: np.ndarray):
    """Calculate the l2 (Euclidean) norm. """
    
    norm = None
    
    # YOUR CODE HERE
    norm = math.sqrt(sum(np.square(v1)))
    
    return norm


# In[6]:


# Autograder test


# In[7]:


def l1_norm(v1: np.ndarray):
    """Calculate the l1 (Manhattan) norm for a given vector."""
    
    norm = None
    
    # YOUR CODE HERE
    norm = sum(abs(v1))
    
    return norm


# In[12]:


# Autograder test


# Hopefully that was all very straightforward. Now onto something a bit more challenging...
# 
# ### 1.3 Normalization (7 points)

# In[13]:


def normalize(v1: np.ndarray):
    """Normalize a non-zero 1d vector. Length is defined here as its l2 norm."""
    
    norm_vector = None
    
    # YOUR CODE HERE
    norm_vector = v1 / l2_norm(v1)
    
    return norm_vector


# In[14]:


# Autograder test


# ### 1.4 Orthogonal Projection (10 points)

# In[15]:


def orth_projection(a1: np.ndarray, a2: np.ndarray):
    "Calculate the orthogonal projection of column vector a2 onto line spanned by column vector a1."
    
    proj = None
    
    # YOUR CODE HERE
    orig_shape = a1.shape
#     proj = ((dot(a2.T, a1)) / (l2_norm(v1)**2)) * a1
    a1 = np.array(a1).flatten()
    a2 = np.array(a2).flatten()
    proj = (dot(a2, a1) / dot(a1, a1)) * a1
    return proj.reshape(orig_shape)


# In[16]:


v1 = np.array([[2, -2, -1]]).T
v2 = np.array([[4, 2, 0]]).T
stu_ans = orth_projection(v1, v2)
stu_ans


# In[17]:


# Student Check

v1 = np.array([[2, -2, -1]]).T
v2 = np.array([[4, 2, 0]]).T
stu_ans = orth_projection(v1, v2)

assert stu_ans.shape==(3, 1), f"Your matrix shape should be (3, 1), not {stu_ans.shape}."
assert np.all(stu_ans==np.array([[8/9, -8/9, -4/9]]).T), """Q1.4, Orthogonal Projection Student Test - 
Your answer is incorrect. You can review Lecture 3 - Orthogonal Projections, Part 1."""

del stu_ans

# Autograder test


# ### 1.5 Gram Schmidt (25 points)

# can use previous functions (normalize & orthogonal projections) to solve 
# 
# 
# solve for u1/u2/u3 and stack vectors at the end
# 
# loop through 
#  - use orthogonal projection
#  - use normalization

# In[18]:


def gram_schmidt(m1: np.ndarray):
    """Implement Gram-Schmidt where the set of vectors are stored as column vectors in m1. You can assume that no vector 
    will be the zero vector for this problem."""
    
    gram_schmidt = None
    
    # YOUR CODE HERE
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


# In[19]:


m1 = np.array([[1, -2, 2], [0, -1, -3], [1, 0, 0]], dtype=np.float64).T
gram_schmidt(m1)


# In[20]:


# Student test

m1 = np.array([[2, -2, -1], [4, 2, 0], [3, 0, 4]], dtype=np.float64).T

stu_ans = gram_schmidt(m1)
assert stu_ans.shape==(3, 3), f"Your matrix shape should be (3, 3), not {stu_ans.shape}."
assert np.all(np.isclose(stu_ans, np.array([[0.6666666666666666, -0.6666666666666666, -0.3333333333333333], 
                 [0.728810888813495, 0.6767529681839596, 0.1041158412590707], 
                 [0.15617376188860604, -0.31234752377721214, 0.9370425713316364]]).T)), """Q1.5, Gram Schmidt, Student 
                 Test - Your answer is incorrect. You can review Lecture 5 - Gram Schmidt Orthogonalisation."""

del stu_ans


# Autograder test


# ## Part 2: Word2Vec (15 points)
# 
# One particular application that uses vectors is textual analysis. Words can be converted into a kind of mathematical representation, e.g., vectors, whose high dimensionality can preserve semantic and syntactic relationships between other words. A word that shares a common context with another word should be in close proximity to each other; quantifying this  similarity in this space can be measured in multiple ways but one of the most common is via cosine similarity.  
# 
# For this exercise, we will be working with a subset of vectorized words already generated from a pre-trained model. The set comes from the 'word2vec-google-news-300' dataset (available in the gensim package); feel free to explore the full data set on your own. 

# In[21]:


import pickle

with open('assets/madwords', 'rb') as f:
    words = pickle.load(f)


# Here we can see words represented as high dimensionality vectors:

# In[22]:


# uncomment to view - yes, that's right, the vector for pickle for the words we just unpickled: 

# print(words['pickle'])


# In[23]:


# words.keys()
len(words['consolidating'])


# In[24]:


# words.get('consolidating')


# In[25]:


# len(words.values())


# In[26]:


# for w in words:
#     print(words[w])


# In[27]:


# words['prince']


# ### 2.1 Cosine Similarity (5 points)

# In[28]:


def cosine_similarity(v1: np.ndarray, v2: np.ndarray):
    """Calculate the cosine similarity between two vectors.  Scipy is also not made available for you here as well."""
    
    cosine_similarity = None
    
    # YOUR CODE HERE
    cosine_similarity = dot(((v1 / l2_norm(v1)).T), (v2 / l2_norm(v2)))
    return cosine_similarity


# In[30]:


# Student test

v1 = np.array([3, 4])
v2 = np.array([5, 12])

stu_ans = cosine_similarity(v1, v2)

assert np.isclose(stu_ans, 0.9692307692307692), """Q2.1, Cosine Similarity, Student 
Test - Your answer is incorrect. You can review Lecture 3 - Orthogonal Projections, Part 1."""

del stu_ans

# Autograder test


# In[31]:


# cosine_similarity(words['emperor'], words.keys())
cosine_similarity(words['emperor'], words['Emperor'])


# ### 2.2 Most Similar Words (10 points)

# In[32]:


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
    
    # YOUR CODE HERE
    if len(list_of_word_vectors) > 1:
        curr_vec = \
        np.mean([np.array(words[word]) for word in list_of_word_vectors], axis=0)
        most_similar_words = calc_cos_sim_between_input_word_and_model_words(curr_vec, model, most_similar_words)
    else:
        curr_vec = model[list_of_word_vectors[0]]
        most_similar_words = calc_cos_sim_between_input_word_and_model_words(curr_vec, model, most_similar_words)
        
    most_similar_words = [tup[0] for tup in sorted(most_similar_words, key=lambda x: x[1], reverse=True)[:10]]

    return tuple(most_similar_words)
    


# In[33]:


# Student test

with open('assets/madwords', 'rb') as f:
    words = pickle.load(f)
    
list_of_words = ['emperor']
stu_ans = most_similar(list_of_words, words)

assert (stu_ans[0]=='emperor'), """Your first word in this list should be 'emperor.' After all, a word is most similar
to itself!"""
assert (stu_ans==('emperor', 'Emperor', 'imperial', 'monarch', 'prince', 
                  'deity', 'monk', 'princes', 'tyrant', 'ruler')), """Q2.2, Most Similar - Your answer is incorrect."""

del stu_ans

# Autograder test


# What happens when you pass in a word that _might_ have different meanings in multiple contexts? Well, let's see:

# In[34]:


list_of_words = ['haircut']
most_similar(list_of_words, words)


# Seems to mostly make sense, right? 'haircut' by itself would largely center around the act of cutting hair. Now what happens when we change the context of 'haircut' when we add another word?

# In[35]:


list_of_words = ['haircut', 'interbank']
most_similar(list_of_words, words)


# Now most closely related words change quite a bit! 'haircut' in the context of the financial markets (particularly related to short-term lending, e.g., the repo and reverse repo markets) is now instead related to the discount on an asset from its market value for a repo rather than the act of cutting hair.

# In[ ]:




