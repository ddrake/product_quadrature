"""
    disk.py

    This script computes the Gaussian weights and points for the disk
    with radius r.
"""
import sympy as sym
from sympy.abc import x
from mpmath import mp, mpf 
from matplotlib import pyplot as plt
import numpy as np
from itertools import product

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


def get_coeffs(n):
    """ Compute recursively pₙ₊₁(x) = (x-aₙ)pₙ(x) + bₙ pₙ₊₁(x)
        with p₀ = 1
        a₀ = 0, aₙ = (xpₙ, pₙ)/(pₙ, pₙ),
        bₙ = (pₙ, pₙ)/(pₙ₋₁, pₙ₋₁) where all inner products are
        weighted by w(x) = |x|.

        Sympy has trouble with the absolute value functions, so 
        we use the definition of the absolute value to write all
        integrals as sums of two integrals.
    """
    a = []                   # a coefficients (should all be zero)
    b = []                   # b coefficients
    pp = []                  # (p,p) inner products
    xpp = []                 # (xp,p) inner products
    p = [mpf(1)]             # p₀ = 1
    #w = sym.Abs(x)          # weight (Jacobian) >= 0 for x in [-1,1]

    mu_0 = sym.integrate(-x,(x,mpf(-1),mpf(0))) + \
           sym.integrate( x,(x,mpf(0),mpf(1))) # should be 1.0

    xppi = sym.integrate(-x*x*p[-1]*p[-1], (x, mpf(-1), mpf(0))) + \
               sym.integrate(x*x*p[-1]*p[-1], (x, mpf(0), mpf(1)))
    xpp.append(mp.chop(xppi,tol=1.0e-48))
    ppi = sym.integrate(-x*p[-1]*p[-1], (x, mpf(-1), mpf(0))) + \
                sym.integrate(x*p[-1]*p[-1], (x, mpf(0), mpf(1)))
    pp.append(mp.chop(ppi,tol=1.0e-48))
    a.append(xpp[-1]/pp[-1])
    p.append((x-a[0])*p[-1])
    for i in range(n):
        xppi = sym.integrate(-x*x*p[-1]*p[-1], (x, mpf(-1), mpf(0))) + \
                   sym.integrate(x*x*p[-1]*p[-1], (x, mpf(0), mpf(1)))
        xpp.append(mp.chop(xppi,tol=1.0e-48))
        ppi = sym.integrate(-x*p[-1]*p[-1], (x, mpf(-1), mpf(0))) + \
                    sym.integrate(x*p[-1]*p[-1], (x, mpf(0), mpf(1)))
        pp.append(mp.chop(ppi,tol=1.0e-48))
        a.append(xpp[-1]/pp[-1])
        b.append(pp[-1]/pp[-2])
        p.append(sym.expand((x-a[-1])*p[-1]-b[-1]*p[-2]))
    return a, b, mu_0, p, pp, xpp


def jacobi(a, b, n):
    """ Construct the modified Golub-Welsh jacobi matrix 
        from coefficients aₙ and bₙ
        for the disk integral.
    """
    b = [mp.sqrt(bb) for bb in b[:n-1]]
    a = a[:n]        # diagonals start with a₀
    j = mp.zeros(n)
    j += mp.diag(a)
    j[1:, :-1] += mp.diag(b)
    j[:-1, 1:] += mp.diag(b)
    return j


def rhos_wts(jacobi, mu_0):
    """ Given a jacobi matrix, return the 
        n points (eigenvalues) and corresponding weights 
        (first terms of normalized eigenvectors)
        of the (modified) Golub-Welsh Jacobi matrix. 
    """
    ew, ev = mp.eigsy(jacobi)
    rhos = [mp.chop(e, tol=1e-30) for e in ew]
    wts = [mp.chop(mu_0*c*c, tol=1.0e-30) for c in ev[0, :]]
    return rhos, wts

def rs(rhos, r):
    """ Given the rho values (based on unit radius) and true radius, 
    return the radii for the quadrature
    """
    return [rho*r for rho in rhos]

def rdjust(num):
    """ Ensure that each number has exactly 26 significant digits
        try to make pretty, given that constraint
    """
    tmp = mp.nstr(num, 26)
    tmp += '0'*(27 - len(tmp) + tmp.index('.'))
    return tmp.rjust(29)

def plot(n, r):
    """ Plot the integration points 
    """
    a,b,mu_0,*rest = get_coeffs(n)
    J = jacobi(a, b, n)
    rhos, _ = rhos_wts(J, mu_0)
    rr = rs(rhos, r)
    tt = [GC1_p(i, n) for i in range(1, n+1)]
    qq = [mp.sqrt(mp.mpf(1)-t*t) for t in tt]
    # matplotlib doesn't like negative radii
    polar = [(mp.asin(tt[j]) + (0 if rr[i] >= 0 else mp.pi), 
              rr[i] if rr[i] >= 0 else -rr[i]) \
                      for i, j in product(range(n), range(n))]
    thetapts, rpts = zip(*polar)
    rrr = np.ones(100)
    th = np.linspace(0, 2*np.pi, 100)

    ax = plt.subplot(111, projection='polar')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.plot(th, r*rrr)
    plt.plot(thetapts, rpts, 'ro')
    plt.title("Disk quadrature points for n={}, r={}".format(n, r))
    plt.show()


def generate_for(a, b, mu_0, n):
    """ Generates the initial data block for one polynomial degree
        Note that this intentionally stores rho values (not converted to r)
    """
    J = jacobi(a, b, n)
    rhos, wts = rhos_wts(J, mu_0)
    tt = [GC1_p(i, n) for i in range(1, n+1)]
    qq = [mp.sqrt(mp.mpf(1)-t*t) for t in tt]
    print('disk_pars[%d] = { %s,' % (n, rdjust(mp.pi / n)))
    print('  {')
    for i in range(n-1):
        print('   { %s, %s, %s, %s },' % (
            rdjust(rhos[i]), rdjust(tt[i]), rdjust(qq[i]), rdjust(wts[i])))
    print('   { %s, %s, %s, %s }' % (
        rdjust(rhos[n-1]), rdjust(tt[n-1]), rdjust(qq[n-1]), rdjust(wts[n-1])))
    print('  }')
    print('};')


def generate(to_m):
    """ Generates the initial data used in DiskGaussQuad in the 
        fiberamp project
    """
    a, b, mu_0, *extras = get_coeffs(to_m)
    for i in range(1, to_m+1):
        generate_for(a, b, mu_0, i)

# ---------------
# test functions
# ---------------
def f1(x, y):
    return x*x*y*y


def f2(x, y):
    return x*y


def f3(x, y):
    return x*y*y


def f4(x, y):
    return x


def f5(x, y):
    return 1


def f6(x, y):
    return x*x


def f7(x, y):
    return y


def f8(x, y):
    return y*y


def f9(x, y):
    return x*x*y*y*y*y


def integrate(f, n, r):
    """ Integrate the given function with the specified
        polynomial degree and radii
    """
    a, b, mu_0, *rest = get_coeffs(n)
    J = jacobi(a, b, n)
    rho, wt = rhos_wts(J, mu_0)
    rr = rs(rho, r)
    tt = [GC1_p(i, n) for i in range(1, n+1)]
    qq = [mp.sqrt(mp.mpf(1)-t*t) for t in tt]

    cwt = mp.pi / n
    result = 0
    for i in range(n):
        for j in range(n):
            result += wt[i]*cwt*f(rr[i]*qq[j], rr[i]*tt[j]) 
    return result * r * r
