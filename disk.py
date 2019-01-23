# The purpose of this script is to get a better understanding of
# how an integral can be computed on a disk using the product formula

import numpy as np
from scipy.linalg import eigh_tridiagonal
from itertools import product
from mpmath import mp
from matplotlib import pyplot as plt

mp.dps = 50

def GC1_p(i, M):
    """ The points for the Gauss-Chebyshev quadrature can be computed
        directly.  These are the t values in (-1, 1) where t = sin(theta)
        for theta in [-pi/2, pi/2]
    """
    return mp.chop(mp.cos(mp.pi * (2*i - 1)/2/M),tol=1.0e-30)


def GC1_w(i, M):
    """ i is included for consistency (not efficient)
    """
    return mp.chop(mp.pi / M,tol=1.0e-30)


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
            0.9546790248493448767148503],
        8: [-0.9646596061808674528345806, -0.8185294874300058668603761,
            -0.5744645143153507855310459, -0.2634992299855422962484895,
            0.2634992299855422962484895, 0.5744645143153507855310459,
            0.8185294874300058668603761, 0.9646596061808674528345806],
        9: [-0.9710282199223060261836893, -0.8503863747508400503582112,
            -0.6452980455813291706201889, -0.3738447061866471744516959,
            0.0000000000000000000000000, 0.3738447061866471744516959,
            0.6452980455813291706201889, 0.8503863747508400503582112,
            0.9710282199223060261836893],
        10: [-0.9762632447087885713212574, -0.8770602345636481685478274,
             -0.7071067811865475244008444, -0.4803804169063914437972190,
             -0.2165873427295972057980989, 0.2165873427295972057980989,
             0.4803804169063914437972190, 0.7071067811865475244008444,
             0.8770602345636481685478274, 0.9762632447087885713212574],
        11: [-0.9798929242261785296900647, -0.8955370355972955669596880,
             -0.7496833930084178248529754, -0.5518475574344457542797839,
             -0.3139029878781443299110139, 0.0000000000000000000000000,
             0.3139029878781443299110139, 0.5518475574344457542797839,
             0.7496833930084178248529754, 0.8955370355972955669596880,
             0.9798929242261785296900647],
        12: [-0.9829724091252897490485174, -0.9113751660173390180078701,
             -0.7869622564275865284525289, -0.6170011401597257548496886,
             -0.4115766110542091326058875, -0.1837532119404283667441428,
             0.1837532119404283667441428, 0.4115766110542091326058875,
             0.6170011401597257548496886, 0.7869622564275865284525289,
             0.9113751660173390180078701, 0.9829724091252897490485174]
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
            0.1102311055883841876377392],
        8: [0.0869637112843634643432660, 0.1630362887156365356567340,
            0.1630362887156365356567340, 0.0869637112843634643432660,
            0.0869637112843634643432660, 0.1630362887156365356567340,
            0.1630362887156365356567340, 0.0869637112843634643432660],
        9: [0.0718567803956129706617061, 0.1406780075747310300960863,
            0.1559132614878706270409275, 0.1115519505417853722012801,
            0.0400000000000000000000000, 0.1115519505417853722012801,
            0.1559132614878706270409275, 0.1406780075747310300960863,
            0.0718567803956129706617061],
        10: [0.0592317212640472718785660, 0.1196571676248416170103229,
             0.1422222222222222222222222, 0.1196571676248416170103229,
             0.0592317212640472718785660, 0.0592317212640472718785660,
             0.1196571676248416170103229, 0.1422222222222222222222222,
             0.1196571676248416170103229, 0.0592317212640472718785660],
        11: [0.0503970963133702100523002, 0.1042253335779769347398516,
             0.1302316957973937456425574, 0.1213467971172424790399570,
             0.0799101883051277416364450, 0.0277777777777777777777778,
             0.0799101883051277416364450, 0.1213467971172424790399570,
             0.1302316957973937456425574, 0.1042253335779769347398516,
             0.0503970963133702100523002],
        12: [0.0428311230947925862600740, 0.0901903932620346518924584,
             0.1169784836431727618474676, 0.1169784836431727618474676,
             0.0901903932620346518924584, 0.0428311230947925862600740,
             0.0428311230947925862600740, 0.0901903932620346518924584,
             0.1169784836431727618474676, 0.1169784836431727618474676,
             0.0901903932620346518924584, 0.0428311230947925862600740]
    }
    return d[M][i-1]


def to_cartesian(r, t, R):
    """ convert from polar to Cartesian coordinates, scaling by the
        disk radius
    """
    return mp.sqrt(1-t*t)*R*r, mp.mpc(R)*r*t

# Use precomputed points and weights
def get_points(M, R=1):
    pts = [(GL_p(i, M), GC1_p(i, M)) for i in range(1, M+1)]
    return [to_cartesian(r, t, R) for r, t in product(*zip(*pts))]


def get_weights(M):
    wts = [(GL_w(i, M), GC1_w(i, M)) for i in range(1, M+1)]
    return [w1*w2 for w1, w2 in product(*zip(*wts))]

# -------------------------------------------
# functions for finding r points and weights
# -------------------------------------------


def get_bs(n):
    """ It turns out that the b values are trivial to compute.
        Get the first n nontrivial b values: 1/2, 1/6, 1/3, 1/5, 3/10, ...
        This sequence is surprisingly simple, but difficult to identify.
        In previous commits I used sympy and scipy to compute polynomials
        recursively and integrate to get norms squared weighted by |r|.
        TODO: Prove that this sequence generates the correct b values for 
        the Gauss-Legendre quadrature on the disk and that the a values are zero
    """
    b = []
    if n <= 0:
        return b
    for i in range(1,n+1):
        if i % 2 == 1:
            b.append( (mp.mpf(i)+1)/4/i )
        else:
            b.append( mp.mpf(i)/4/(i+1) )
    return b

def jacobi(n):
    """ Construct the Golub-Welsh jacobi matrix for coefficients b
        for the disk integral.  In this case, the diagonal entries
        (a values) will always be zero.
    """
    b = [mp.sqrt(b) for b in get_bs(n-1)]
    j = mp.zeros(n)
    j[1:,:-1] += mp.diag(b)
    j[:-1,1:] += mp.diag(b)
    return j

def rs_ts(n):
    """ Given n, the number of 1D radii, return the 
        n points (eigenvalues) and corresponding weights 
        (first terms of normalized eigenvectors)
        of the Golub-Welsh Jacobi matrix. 
        These should match the values hard coded in GL_p and GL_w
    """
    ew, ev = mp.eigsy(jacobi(n))
    return [mp.chop(e, tol=1e-30) for e in ew], [mp.chop(c*c,tol=1.0e-30) for c in ev[0, :]]

# Compute directly points and weights
def get_points1(M, R=1):
    rs, _ = rs_ts(M)
    pts = [(rs[i-1], GC1_p(i, M)) for i in range(1, M+1)]
    return [to_cartesian(r, t, R) for r, t in product(*zip(*pts))]


def get_weights1(M):
    _, ts = rs_ts(M)
    wts = [(ts[i-1], GC1_w(i, M)) for i in range(1, M+1)]
    return [w1*w2 for w1, w2 in product(*zip(*wts))]

def rdjust(num):
    """ Ensure that each number has exactly 26 significant digits
        try to make pretty, given that constraint
    """
    tmp = mp.nstr(num,26)
    tmp += '0'*(27 - len(tmp) + tmp.index('.'))
    return tmp.rjust(29)

def plot_for(M):
    """ Plot the integration points 
    """
    rs, _ = rs_ts(M)
    tt = [GC1_p(i, M) for i in range(1, M+1)]
    polar = [(mp.asin(tt[j]), rs[i]) for i,j in product(range(M), range(M))]
    # work around the fact that matplotlib can't handle negative r values
    polar = [(theta,r) if r>=0 else (theta+mp.pi,-r) for theta, r in polar]
    thetapts,rpts = zip(*polar)
    r = np.ones(100)
    th = np.linspace(0,2*np.pi,100)

    ax = plt.subplot(111, projection='polar')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.plot(th, r)
    plt.plot(thetapts, rpts, 'ro')
    plt.title("Unit Disk points for n={}".format(M))
    plt.show()

def generate_for(n):
    """ Generates the initial data for the C++ structs in the fiberamp project
    """
    rs, wts = rs_ts(n)
    tt = [GC1_p(i, n) for i in range(1,n+1)]
    qq = [mp.sqrt(mp.mpf(1)-t*t) for t in tt]
    print('disk_pars[%d] = { %s,' % (n, rdjust(mp.pi / n)))
    print('  {')
    for i in range(n-1):
        print('   { %s, %s, %s, %s },' % (rdjust(rs[i]), rdjust(tt[i]), rdjust(qq[i]), rdjust(wts[i])))
    print('   { %s, %s, %s, %s }' % (rdjust(rs[n-1]), rdjust(tt[n-1]), rdjust(qq[n-1]), rdjust(wts[n-1])))
    print('  }')
    print('};')

def generate(to_m):
    for i in range(1,to_m+1):
        generate_for(i)


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
    return mp.exp((x-.1)*mp.sin(y-.2))

def f7(x, y):
    return x*x*y*y*y*y

def integrate(f, M, R=1.0):
    """ M-degree Gaussian Product Approximation to integral of f on the 
        disk centered at the origin with radius R
        seems to be accurate to about 16 decimal places
    """
    pts = get_points(M, R)
    wts = get_weights(M)
    return R*R*sum((f(*pt)*wt) for pt, wt in zip(pts, wts))

def integrate1(f, M, R=1.0):
    """ Same as integrate but computing points and weights instead of 
        using pre-computed values.  Seems to be accurate to about 15 decimals
    """
    pts = get_points1(M, R)
    wts = get_weights1(M)
    return R*R*sum((f(*pt)*wt) for pt, wt in zip(pts, wts))
