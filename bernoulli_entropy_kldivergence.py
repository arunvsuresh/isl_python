import numpy as np
import math
from typing import List


# ### 1.1. Counting Principles
def items_in_sets(items: List) -> int:
    """Given a list with $n$ elements. Each element represents a set and its value represents 
    how many objects are in that set. For example, a list of [1, 5] would mean there are 2 sets,  
    containing 1 and 5 objects respectively. If no two objects are the same, how many ways are there to 
    select a single item from the union of all sets?"""
    
    items_in_sets = None
    
    # 
    items_in_sets = sum(items)
    return items_in_sets

# ### 1.2 Counting Principles, continued
def items_repeat(n: int, k: int) -> int:
    """
    Given n distinct letters, how many ways can you create a k-length string using those 
    letters, assuming you can repeat letters in the string? Assume that n and k are greater than 0.
    """
    
    items_repeat = None
    
    # 
    items_repeat = n**k
    
    return items_repeat

# ### 1.3 Counting Principles, continued

def items_no_repeat(n: int, k: int) -> int:
    """
    Given n distinct letters, how many ways can you create a k-length string using those letters, 
    assuming you cannot repeat letters in the string? Assume that n and k are greater than 0.
    """
    
    items_repeat = None
    
    # 
#     n_minus_k = n - k
    if n >= k:
        items_repeat = math.factorial(n) / math.factorial(n - k)
    else:
        items_repeat = 0
    return items_repeat

# ### 1.4 Counting Principles, continued
def meal_combos(a: int, m: int, d: int, num: int) -> int:
    """
    Given a number of appetizers, m number of main dishes, and d number of desserts, how many 
    different ways can you form a meal of num number of  appetizers, num number of mains, and 
    num number of  desserts? 
    
    E.g., if a=10, m=11, d=12 and num=1, then you are to find how many different ways you can 
    combine 1 appetizer out of 10 appetizers, 1 main out of 11 mains, and 1 dessert out of 
    12 desserts. 
    
    Assume that a, m, d and num are greater than 0. 
    """
    
    meal_combos = None
    
    def combo(n, k):
        if n >= k:
            return math.factorial(n) / (math.factorial(n - k) * math.factorial(k))
        else:
            return 0
    # 
#     1 choose 10, 
    apps, mains, desserts = \
    combo(a, num), combo(m, num), combo(d, num)
    
    meal_combos = apps * mains * desserts
    return meal_combos

# ### 1.5 Bernoulli and Binomial Distributions
def probability_event() -> float:
    """
    Let's say the probability of failing this math class is .28 (and passing the class is .72). 
    Given 10 students in the class, what is the probability of exactly 2 students failing? 
    
    """
    
    probability_event = None
    
    # 
    p = .28
    n, k = 10, 2
    n_choose_k = math.comb(n, k)
    probability_event = round(n_choose_k * (p ** k) * ((1 - p) ** (n - k)), 3)
    
    
    return probability_event

# ## 1.6 Bernoulli and Binomial Distributions
def probability_event_odd(k: int) -> float:
    """
    Let's keep the same probabilities as 
    before (.28 probability of failing the class, .72 probability of passing), but this time the number 
    of students are unknown. What is the probability that an odd number of students fail? 
    
    """
    
    probability_event_odd = None
    
    # 
    p = .28
    probability_event_odd = 0
    for i in range(1, k + 1, 2):
        n_choose_k = math.comb(k, i)
        probability_event_odd += n_choose_k * (p ** i) * ((1 - p) ** (k - i))
    return round(probability_event_odd, 3)
k = 2
stu_ans = probability_event_odd(2)

assert stu_ans == .403, """Q1.6, Bernoulli and Binomial Distributions - Your answer is incorrect."""

# ## 1.7 Entropy and KL Divergence
def information_content(p: float) -> float:
    """Given a discrete random variable with a probability of an event being p, what is the information content
    contained by this event? Assume log is base 2. Round your answer to the nearest 3 digits before returning.
    """
    
    information_content = None
    
    # 
    information_content = -np.log2(p)
    information_content = round(information_content, 3)
    return information_content

k = .5
stu_ans = information_content(k)

assert stu_ans == 1, """Q1.7, Entropy and KL Divergence - Your answer is incorrect."""

# ## 1.8 Entropy and KL Divergence, continued

def entropy(pmf: List) -> float:
    """
    Given an pmf of a discrete random variable X, calculate its entropy. 
    
    Assume in this case, log is base e.
    """
    
    entropy = None
    
    # 
    entropy = 0
    for prob in pmf:
        if prob != 0:
            entropy += (prob * np.log(prob))
    entropy = -1 * entropy
    return round(entropy, 3)

test = [.1, .15, .7, .05, 0]

stu_ans = entropy(test)

assert stu_ans == .914, """Q1.8, Entropy and KL Divergence - Your answer is incorrect. """

# ## 1.9 Entropy and KL Divergence, continued
def kullback_leibler_divergence(pmf: List, estimated_pmf: List) -> float:
    """
    Given a known pmf of a discrete random variable and a count of samples drawn in order to 
    estimate a pmf (the estimated pmf), calculate the relative entropy between the two. 
    
    Assume base e for all log calculations.
    """
    
    kl_divergence = None
    
    # 
    total_samples = sum(estimated_pmf)
    q_x = [counts / total_samples for counts in estimated_pmf]
    p_x = pmf
    kl_divergence = 0
    for px, qx in list(zip(p_x, q_x)):
        if qx != 0:
            kl_divergence += px * np.log(px / qx)

    return round(kl_divergence, 3)

pmf = [.1, .15, .7, .05, 0]
estimated_count = [60, 69, 346, 25, 0]

stu_ans = kullback_leibler_divergence(pmf, estimated_count)

assert stu_ans == .002, """Q1.9, Entropy and KL Divergence - Your answer is incorrect. """

# ## 1.10 Entropy and KL Divergence, continued
def mutual_information(matrix: np.array) -> float:
    """
    Given a matrix of probabilities of two discrete random variables, X and Y, calculate the mutual 
    information. The (i,j)-th entry (i.e. the entry in the i-th row and j-th column) in the matrix represents 
    the joint probability that X = x_j and Y = y_i.
    
    Assume base e for all log calculations. 
    """
    
    mi = None
    
    # 
    mi = 0
    px = matrix[0]
    py = matrix[1]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            joint_prob = matrix[i, j]
            if joint_prob != 0 and px[j] != 0 and py[i] != 0:
                mi += joint_prob * np.log(joint_prob / (px[j] * py[i]))
            
    
    return round(mi, 3)

xy = np.array([
               [.20, .10 ], 
               [.40, .30]]
             )
mutual_information(xy)

xy = np.array([[.20, .10 ], [.40, .30]])
stu_ans = mutual_information(xy)

assert stu_ans == .004, """Q1.10, Entropy and KL Divergence - Your answer is incorrect."""


