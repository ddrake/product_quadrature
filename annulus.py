"""
    annulus.py

    This script computes the Gaussian weights and points for the annulus
    with radii r1 and r2.  If r1 = 0 and r2 = 1, this computes an
    alternatve Gaussian quadrature for the unit disk.
"""
import sympy as sym
from sympy.abc import x
from mpmath import *
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


def get_coeffs(n, r1=0, r2=1):
    """ Compute recursively pₙ₊₁(x) = (x-aₙ)pₙ(x) + bₙ pₙ₊₁(x)
        with p₀ = 1, p₁ = x
        a₀ = 0, aₙ = (xpₙ, pₙ)/(pₙ, pₙ),
        bₙ = (pₙ, pₙ)/(pₙ₋₁, pₙ₋₁) where all inner products are
        weighted by w(x) = x + c.
        For the annulus, we always have c = (r2 + r1) / (r2 - r1) >= 1.
        When r1 = 0, and r2 = 1, then c = 1, 
        and this then provides an alternative quadrature
        of the disk.
    """
    r1, r2 = mpf(r1), mpf(r2)
    gamma = (r2-r1)/2        # divided difference of radii
    c = (r2+r1)/(r2-r1)      # average radius over divided difference
    a = []                   # a coefficients
    b = []                   # b coefficients
    pp = []                  # (p,p) inner products
    xpp = []                 # (xp,p) inner products
    p = [mpf(1)]             # p₀ = 1
    w = c + x                # weight (Jacobian) >= 0 for x in [-1,1]

    xpp.append(sym.integrate(w*x*p[-1]*p[-1],(x, mpf(-1), mpf(1))))
    pp.append(sym.integrate(w*p[-1]*p[-1],(x,mpf(-1),mpf(1))))
    a.append(xpp[-1]/pp[-1])
    p.append((x-a[0])*p[-1]) 
    for i in range(n):
        xpp.append(sym.integrate(w*x*p[-1]*p[-1], (x, mpf(-1), mpf(1))))
        pp.append(sym.integrate(w*p[-1]*p[-1], (x, mpf(-1), mpf(1))))
        a.append(xpp[-1]/pp[-1])
        b.append(pp[-1]/pp[-2])
        p.append(sym.expand((x-a[-1])*p[-1]-b[-1]*p[-2]))
    return a,b,p,pp,xpp

def jacobi(a,b,n):
    """ Construct the modified Golub-Welsh jacobi matrix 
        from coefficients aₙ and bₙ
        for the disk integral.
    """
    b = [mp.sqrt(bb) for bb in b[:n-1]]
    a = a[:n]        # diagonals start with a₀
    j = mp.zeros(n)
    j += mp.diag(a)
    j[1:,:-1] += mp.diag(b)
    j[:-1,1:] += mp.diag(b)
    return j

def rhos_wts(jacobi):
    """ Given a jacobi matrix, return the 
        n points (eigenvalues) and corresponding weights 
        (first terms of normalized eigenvectors)
        of the (modified) Golub-Welsh Jacobi matrix. 
    """
    ew, ev = mp.eigsy(jacobi)
    rhos = [mp.chop(e, tol=1e-30) for e in ew]
    wts = [mp.chop(c*c,tol=1.0e-30) for c in ev[0, :]]
    return rhos, wts

def rs(rhos, r1, r2):
    """ Given the rho values and the inner and outer radii,
        return the radial points for the quadrature.
    """
    return [(r2-r1)/2*rho + (r2+r1)/2 for rho in rhos]

def rdjust(num):
    """ Ensure that each number has exactly 26 significant digits
        try to make pretty, given that constraint
    """
    tmp = mp.nstr(num,26)
    tmp += '0'*(27 - len(tmp) + tmp.index('.'))
    return tmp.rjust(29)

def plot_for(a, b, n, r1, r2):
    """ Plot the integration points 
    """
    J = jacobi(a, b, n)
    rhos, wts = rhos_wts(J)
    r = rs(rhos, r1, r2)
    tt = [GC1_p(i, n) for i in range(1,n+1)]
    qq = [mp.sqrt(mp.mpf(1)-t*t) for t in tt]
    cart1 = [(r[i]*qq[j], r[i]*tt[j]) for i,j in product(range(n), range(n))]
    cart2 = [(-r[i]*qq[j],-r[i]*tt[j]) for i,j in product(range(n), range(n))]
    cart = cart1+cart2
    x,y = zip(*cart)
    th = np.linspace(0,2*np.pi,100)
    xout, yout = r2*np.cos(th), r2*np.sin(th)
    xin, yin = r1*np.cos(th), r1*np.sin(th)

    plt.axis('equal')
    plt.plot(xin,yin)
    plt.plot(xout,yout)
    plt.axis([-r2*1.2, r2*1.2, -r2*1.2, r2*1.2])
    plt.plot(x,y, 'ro')
    plt.show()

def generate_for(a, b, n, r1, r2):
    """ Generates the initial data for the C++ structs in the fiberamp project
        I think that to obtain symmetry, we need 
    """
    J = jacobi(a, b, n)
    rhos, wts = rhos_wts(J)
    r = rs(rhos, r1, r2)
    r += [-rr for rr in r]
    wts += wts[:]
    tt = [GC1_p(i, n) for i in range(1,n+1)]
    qq = [mp.sqrt(mp.mpf(1)-t*t) for t in tt]
    print('disk_pars[%d] = { %s,' % (n, rdjust(mp.pi / n)))
    print('  {')
    for i in range(n-1):
        print('   { %s, %s, %s, %s },' % (rdjust(r[i]), rdjust(tt[i]), rdjust(qq[i]), rdjust(wts[i])))
        print('   { %s, %s, %s, %s },' % (rdjust(r[i]), rdjust(-tt[i]), rdjust(-qq[i]), rdjust(wts[i])))
    print('   { %s, %s, %s, %s },' % (rdjust(r[n-1]), rdjust(tt[n-1]), rdjust(qq[n-1]), rdjust(wts[n-1])))
    print('   { %s, %s, %s, %s }' % (rdjust(r[n-1]), rdjust(-tt[n-1]), rdjust(-qq[n-1]), rdjust(wts[n-1])))
    print('  }')
    print('};')

def generate(to_m, r1, r2):
    a, b, *extras = get_coeffs(to_m, r1, r2)
    for i in range(1,to_m+1):
        generate_for(a, b, i, r1, r2)

def f1(x,y):
    return x*x*y*y

def f2(x,y):
    return x*y

def f3(x,y):
    return x*y*y

def f4(x,y):
    return x

def f5(x,y):
    return 1

def f6(x,y):
    return x*x

def f7(x,y):
    return y

def f8(x,y):
    return y*y

def integrate(f, n, r1, r2):
    a, b, *rest = get_coeffs(n, r1, r2)
    J = jacobi(a, b, n)
    rho, wt = rhos_wts(J)
    r = rs(rho, r1, r2)
    tt = [GC1_p(i, n) for i in range(1,n+1)]
    qq = [mp.sqrt(mp.mpf(1)-t*t) for t in tt]

    cwt = mp.pi / n
    result = 0
    for i in range(n):
        for j in range(n):
            print('r', r[i])
            print('(x,y)', r[i]*qq[j], r[i]*tt[j])
            print('(x1,y1)', -r[i]*qq[j], -r[i]*tt[j])
            print('f(x,y)', f(r[i]*qq[j], r[i]*tt[j]))
            print('f(x1,y1)', f(-r[i]*qq[j], -r[i]*tt[j]))
            print('wt',wt[i],'cwt',cwt)
            result += wt[i]*cwt*f(r[i]*qq[j], r[i]*tt[j]) \
                    + wt[i]*cwt*f(-r[i]*qq[j], -r[i]*tt[j])
    return result * (r2-r1) * (r2-r1) / 4


def g1(r):
    return 1

def g2(r):
    return r

def g3(r):
    return r*r

def integrate1d(g, n, r1, r2):
    a, b, *rest = get_coeffs(n, r1, r2)
    J = jacobi(a, b, n)
    rho, wt = rhos_wts(J)
    r = rs(rho, r1, r2)
    result = 0
    for i in range(n):
        for j in range(n):
            print('r', r[i])
            print('g(r)', g(r[i]))
            print('wt',wt[i])
            result += wt[i]*g(r[i])
    return result * (r2-r1) * (r2-r1) / 4


