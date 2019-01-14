# The purpose of this script is to get a better understanding of
# how an integral can be computed on a disk using the product formula

import numpy as np
from scipy.linalg import eig
from scipy.integrate import quad
from itertools import product
import sympy.abc as sym
from sympy.functions import Abs
import sympy

def GC1_p(i, M):
    """ The points for the Gauss-Chebyshev quadrature can be computed
        directly.  These are the t values in (-1, 1) where t = sin(theta)
        for theta in [-pi/2, pi/2]
    """
    return np.cos((2*i - 1)/2/M * np.pi)


def GC1_w(i, M):
    """ i is included for consistency (not efficient)
    """
    return np.pi / M


def GL_p(i, M):
    """ These are the r values in (-1, 1).
        Because we have the weighting function |r|, we need to solve a 
        three term recurrence p_{r+1}(x) = (x-a_r)p_r(x) - b_r p_{r-1}(x)
        with the convention p_{-1}(x) = 0
        a_r = (xp_r, p_r)/(p_r,p_r), b_r = (p_r, p_r)/(p_{r-1}, p_{r-1})
        where the inner product (f,g) = int_a^b w(x) f(x) g(x) dx
        the points and weights can then be computed using the Golub-Welsch
        algorithm (https://en.wikipedia.org/wiki/Gaussian_quadrature)
        Look at source of scipy.special.orthogonal for help computing these
        Values below were taken from C code by Pavel Holoborodko
        and hand checked by me.
    """
    d = {
        1: [0],
        2: [-0.7071067811865475244008444, 0.7071067811865475244008444],
        3: [-0.8164965809277260327324280, 0.0, 0.8164965809277260327324280],
        4: [-0.8880738339771152621607646, -0.4597008433809830609776340,
            0.4597008433809830609776340, 0.8880738339771152621607646],
        5: [-0.9192110607898045793726291, -0.5958615826865180525340234,
            0.0000000000000000000000000, 0.5958615826865180525340234,
            0.9192110607898045793726291],
        6: [-0.9419651451198933233901941, -0.7071067811865475244008444,
            -0.3357106870197288066698994, 0.3357106870197288066698994,
            0.7071067811865475244008444, 0.9419651451198933233901941],
        7: [-0.9546790248493448767148503, -0.7684615381131740734708478,
            -0.4608042298407784190147371, 0.0000000000000000000000000,
            0.4608042298407784190147371, 0.7684615381131740734708478,
            0.9546790248493448767148503]
    }
    return d[M][i-1]


def GL_w(i, M):
    # weights from the Golub-Welsch algorithm (see above)
    # Values below were taken from C code by Pavel Holoborodko
    d = {
        1: [1.0],
        2: [0.5, 0.5],
        3: [0.375, 0.25, 0.375],
        4: [0.25, 0.25, 0.25, 0.25],
        5: [0.1882015313502336375250377, 0.2562429130942108069194067,
            0.1111111111111111111111111, 0.2562429130942108069194067,
            0.1882015313502336375250377],
        6: [0.1388888888888888888888889, 0.2222222222222222222222222,
            0.1388888888888888888888889, 0.1388888888888888888888889,
            0.2222222222222222222222222, 0.1388888888888888888888889],
        7: [0.1102311055883841876377392, 0.1940967344215859403901162,
            0.1644221599900298719721446, 0.0625000000000000000000000,
            0.1644221599900298719721446, 0.1940967344215859403901162,
            0.1102311055883841876377392]
    }
    return d[M][i-1]


def to_cartesian(r, t, R):
    """ convert from polar to Cartesian coordinates, scaling by the
        disk radius
    """
    return R*r*np.sqrt(1-t*t), R*r*t


def get_points(M, R):
    pts = [(GL_p(i, M), GC1_p(i, M)) for i in range(1, M+1)]
    return [to_cartesian(r, t, R) for r, t in product(*zip(*pts))]


def get_weights(M):
    wts = [(GL_w(i, M), GC1_w(i, M)) for i in range(1, M+1)]
    return [w1*w2 for w1, w2 in product(*zip(*wts))]

# -------------------------------------------
# functions for finding r points and weights
# -------------------------------------------

def p_np1(p_nm1, p_n, b):
    """ use sympy to compute the next Legendre polynomial from the 
        previous two polynomials and the b coefficient.
        Expanding seems to speed things up, but precision is lost 
        more quickly.  May want to remove it.
    """
    return sympy.expand( sym.x * p_n - b * p_nm1)

def get_recip_normsq(p_n):
    """ Return the reciprocal of the square of the norm (weighted by |r|)
        of the given polynomial, computed using the scipy quad integrator. 
    """
    lam1 = sympy.lambdify([sym.x], Abs(sym.x)*p_n*p_n, modules=['math'])
    return 1/quad(lam1,-1,1)[0]


def get_bs_and_ps(n):
    """ There are some interesting features of the b values used in the 
        polynomial recurrence relation for r.  
        Each b value is a ratio of two inner products (p_n, p_n) 
        and (p_nm1, p_nm1), where both inner products are weighted by |r|.
        One obvious but useful fact is that there is no need to compute
        the inner product for the denominator, since it was computed in 
        the previous step.  The second (interesting) fact is that these 
        inner products seem to always result in a reciprocal of an integer,
        (denoted in this function as ri) at least up to n = 13.  
        The first 7 odd numbered terms match OEIS sequence A000515. 
        The reciprocals of the b values are also interesting in that
        successive pairs have the same denominator and are symmetric 
        about and converging to the number 4.  The b values up to 13 
        can thus be computed very precisely as a ratio of two integers.
    """
    ps = [1, sym.x]
    bs = []
    ris = [1]
    errs = []
    i = 1
    while i < n:
        ri = ris[-1]
        newri = get_recip_inner(ps[-1])
        newrir = round(newri)
        err = abs(newrir-newri)
        errs.append(err)
        if err < 1.0e-5:
            newri = newrir
        ris.append(newri)
        b = ri/newri
        bs.append(b)
        ps.append(p_np1(*ps[-2:], b))
        i += 1
    return bs,ps,ris,errs

def jacobi(b):
    """ Construct the Golub-Welsh jacobi matrix for coefficients b
        for the disk integral.  In this case, the diagonal entries
        will always be zero.
    """
    n = len(b)
    b = np.sqrt(np.array(b))
    return np.zeros((n+1, n+1)) + np.diag(b, -1) + np.diag(b, 1)


def rs_ts(j):
    """ Given the Golub-Welsh jacobi matrix, return the
        r points (eigenvalues) and corresponding weights 
        (first terms of normalized eigenvectors)
    """
    ew, ev = eig(j)
    return ew, ev[0, :]**2

# ---------------
# test functions
# ---------------


def f1(x, y):
    return x*y*y


def f2(x, y):
    return x*x*y*y


def f3(x, y):
    return 1


def f4(x, y):
    return x*x


def f5(x, y):
    return y*y


def f6(x, y):
    return np.exp((x-.1)*np.sin(y-.2))


def integrate(f, M, R=1.0):
    """ M-degree Gaussian Product Approximation to integral of f on the 
        disk centered at the origin with radius R
    """
    pts = get_points(M, R)
    wts = get_weights(M)
    return R*R*sum((f(*pt)*wt) for pt, wt in zip(pts, wts))
