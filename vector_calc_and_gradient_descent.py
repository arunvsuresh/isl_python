import numpy as np
import math
from typing import List, Callable

# ### Essential Vector Calculus
def derive_powers(powers: List) -> Callable:
    """
    Given a univariate function represented as a list of powers, create and return its first order 
    derivative as a function. 
    
    Example: 
      powers[9,-2]: this represents the univariate function: 9*x - 2
      powers=[4,2,3]: this represents the univariate function: 4(x**2) + 2*(x) + 3
    """
    
#     derivative = lambda x: x
    
    def derivative(x):
        deriv_val = 0
        i = len(powers) - 1
        for power in powers:
            
            if i == 0:
                return deriv_val
            if i == 1:
                deriv_val += power
            else:
                deriv_val += i * power * (x**(i - 1))
            i -= 1
        
    return derivative

powers = [3, -16, 18, 0, 0]
x = 3
stu_ans = derive_powers(powers)
stu_ans = stu_ans(x)

assert stu_ans==0, """Essential Vector Calculus - Your answer is incorrect."""

# ### 1.2. Gradient of Quadratic Forms

# $ f(x) = x^{T} Ax $.

def gradient_quadratic(m: np.ndarray) -> np.ndarray:
    """
    Given a square n x n matrix A and the function f(x) shown above, 
    calculate the matrix B such that B@x is the gradient of f(x). 
    Set the variable gradient_quadratic equal to B.
    """
    
    gradient_quadratic = None
    
    A = m
    B = A + A.T
    gradient_quadratic = B
    return gradient_quadratic

# ### Necessary Conditions for Optimality

# We are going to work with the univariate function 
# $f(x) = x^{4\ }-2\ x^{3\ }-2\ x^{2\ }+6x\ -12 $

import sympy as sp
def stationary_points() -> List:
    """
    What are the stationary points of f(x)? Return the result as a list of x. 
    Be sure to order the points listed in ascending order. 
    """
    
    stationary_points = None
    

    x = sp.symbols('x')
    f_x = x**4 - 2*x**3 - 2*x**2 + 6*x - 12
    f_prime = sp.diff(f_x, x)

    critical_points = sp.solve(f_prime, x)
    
    stationary_points = sorted(critical_points)

    return stationary_points

stationary_points()

# ###  Gradient Descent, constant step

def univariate_gd_constant(powers: List, 
                           initial_point: float, 
                           step_size: float,
                           num_iterations: int) -> float:
    """
    Given a univariate function represented as a list of powers, an initial starting point, 
    a constant step size, and the number of iterations, implement gradient descent and return 
    x after num_iteration number of iterations.
    
    
    Example: 
      powers[9,-2]: this represents the univariate function: 9*x - 2
      powers=[4,2,3]: this represents the univariate function: 4(x**2) + 2*(x) + 3
      
    """
    
    x = initial_point
    
    x_sym = sp.symbols('x')
    
    f_x = sum(p * x_sym**i for i, p in enumerate(powers[::-1]))
    
    f_prime = sp.diff(f_x, x_sym)
    
    for _ in range(num_iterations):
        gradient = f_prime.subs(x_sym, x).evalf()
        x = x - step_size * gradient
    
    x = round(float(x), 3)
    
    return x

powers = [3, -16, 18, 0, 0]
stu_ans = univariate_gd_constant(powers, -.8, .02, 5)

assert stu_ans == .039, """Gradient Descent - Your answer is incorrect."""

stu_ans = univariate_gd_constant(powers, 3.8, .02, 5)

assert stu_ans == 2.991, f"""Gradient Descent - Your answer should be 2.991 rather than {stu_ans}."""


# ###  Gradient Descent, adjusted step size
def univariate_gd_adjusted(powers: List, 
                           initial_point: float, 
                           step_size: float,
                           decay: float,
                           num_iterations: int) -> float:
    """
    Given a univariate function represented as a list of powers, an initial starting point, 
    a constant step size, a decay factor on the step size, and the number of iterations, 
    implement gradient descent and return x after num_iteration number of iterations.
    
    
    Example: 
      powers[9,-2]: this represents the univariate function: 9*x - 2
      powers=[4,2,3]: this represents the univariate function: 4(x**2) + 2*(x) + 3
      
    """
    
    x = initial_point
    
    x_sym = sp.symbols('x')
    
    f_x = sum(p * x_sym**i for i, p in enumerate(powers[::-1]))
    
    f_prime = sp.diff(f_x, x_sym)
    
    for _ in range(num_iterations):
        gradient = f_prime.subs(x_sym, x).evalf()
        x = x - step_size * gradient
        step_size *= decay  
    
    x = round(float(x), 3)
    
    return x

powers = [3, -16, 18, 0, 0]
stu_ans = univariate_gd_adjusted(powers, -.8, .05, .7, 5)


assert stu_ans == 2.995, f"""Gradient Descent - Your answer should be 2.995, rather than {stu_ans}."""