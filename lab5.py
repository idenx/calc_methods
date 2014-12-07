from __future__ import division
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.colors import LogNorm

class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def to_tuple(self):
        return (self.start, self.end)

class Area2d(object):
    def __init__(self, x_interval, y_interval):
        self.x = x_interval
        self.y = y_interval

def nelder_mead(f, x0, method="ANMS", tol=1e-8, maxit=1e4, iter_returns=None):
    """
    This is a python implementation of the nelder-mead algorithm, the
    goal is to use numba to attempt to speed it up in a simple manner.

    Parameters
    ----------

    f : callable
        Function to minimize
    x0 : scalar(float) or array_like(float, ndim=1)
        The initial guess for minimizing
    method : string or tuple(floats)
        If a string, should specify ANMS or NMS then will use specific
        parameter values, but also can pass in a tuple of parameters in
        order (alpha, beta, gamma, delta), which are the reflection,
        expansion, contraction, and contraction parameters
    tol : scalar(float)
        The tolerance level to achieve convergence
    maxit : scalar(int)
        The maximimum number of iterations allowed


    References :

    Nelder, J. A. and R. Mead, "A Simplex Method for Function
    Minimization." 1965. Vol 7(4). Computer Journal

    F. Gao, L. Han, "Implementing the Nelder-Mead simplex algorithm with
    adaptive parameters", Comput. Optim. Appl.,

    http://www.brnt.eu/phd/node10.html#SECTION00622200000000000000


    TODO:
      * Check to see whether it works with numba
      * Check to see whether we can use an array instead of a list of
      tuples
      * Write some tests
      * Implement in Julia:
      https://github.com/JuliaOpt/Optim.jl/blob/master/src/nelder_mead.jl
    """
    #-----------------------------------------------------------------#
    # Set some parameter values
    #-----------------------------------------------------------------#
    init_guess = x0
    fx0 = f(x0)
    dist = 10.
    curr_it = 0

    # Get the number of dimensions we are optimizing
    n = np.size(x0)

    # Will use the Adaptive Nelder-Mead Simplex paramters by default
    if method is "ANMS":
        alpha = 1.
        beta = 1. + (2./n)
        gamma = .75 - 1./(2.*n)
        delta = 1. - (1./n)
    # Otherwise can use standard parameters
    elif method is "NMS":
        alpha = 1.
        beta = 2.
        gamma = .5
        delta = .5
    elif type(method) is tuple:
        alpha, beta, gamma, delta = method


    #-----------------------------------------------------------------#
    # Create the simplex points and do the initial sort
    #-----------------------------------------------------------------#
    simplex_points = np.empty((n+1, n))

    pt_fval = [(x0, fx0)]

    simplex_points[0, :] = x0

    for ind, elem in enumerate(x0):

        if np.abs(elem) < 1e-14:
            curr_tau = 0.00025
        else:
            curr_tau = 0.05

        curr_point = np.squeeze(np.eye(1, M=n, k=ind)*curr_tau + x0)

        simplex_points[ind, :] = curr_point
        pt_fval.append((curr_point, f(curr_point)))
        
    if iter_returns is not None:
        ret_points = []
    else:
        ret_points = None


    #-----------------------------------------------------------------#
    # The Core of The Nelder-Mead Algorithm
    #-----------------------------------------------------------------#
    while dist>tol and curr_it<maxit:

        # 1: Sort and find new center point (excluding worst point)
        pt_fval = sorted(pt_fval, key=lambda v: v[1])
        xbar = x0*0

        for i in range(n):
            xbar = xbar + (pt_fval[i][0])/(n)
            
        if iter_returns is not None and curr_it in iter_returns:
            ret_points.append(pt_fval)

        # Define useful variables
        x1, f1 = pt_fval[0]
        xn, fn = pt_fval[n-1]
        xnp1, fnp1 = pt_fval[n]


        # 2: Reflect
        xr = xbar + alpha*(xbar - pt_fval[-1][0])
        fr = f(xr)

        if f1 <= fr < fn:
            # Replace the n+1 point
            xnp1, fnp1 = (xr, fr)
            pt_fval[n] = (xnp1, fnp1)

        elif fr < f1:
            # 3: expand
            xe = xbar + beta*(xr - xbar)
            fe = f(xe)

            if fe < fr:
                xnp1, fnp1 = (xe, fe)
                pt_fval[n] = (xnp1, fnp1)
            else:
                xnp1, fnp1 = (xr, fr)
                pt_fval[n] = (xnp1, fnp1)

        elif fn <= fr <= fnp1:
            # 4: outside contraction
            xoc = xbar + gamma*(xr - xbar)
            foc = f(xoc)

            if foc <= fr:
                xnp1, fnp1 = (xoc, foc)
                pt_fval[n] = (xnp1, fnp1)
            else:
                # 6: Shrink
                for i in range(1, n+1):
                    curr_pt, curr_f = pt_fval[i]
                    # Shrink the points
                    new_pt = x1 + delta*(curr_pt - x1)
                    new_f = f(new_pt)
                    # Replace
                    pt_fval[i] = new_pt, new_f

        elif fr >= fnp1:
            # 5: inside contraction
            xic = xbar - gamma*(xr - xbar)
            fic = f(xic)

            if fic <= fr:
                xnp1, fnp1 = (xic, fic)
                pt_fval[n] = (xnp1, fnp1)
            else:
                # 6: Shrink
                for i in range(1, n+1):
                    curr_pt, curr_f = pt_fval[i]
                    # Shrink the points
                    new_pt = x1 + delta*(curr_pt - x1)
                    new_f = f(new_pt)
                    # Replace
                    pt_fval[i] = new_pt, new_f

        # Compute the distance and increase iteration counter
        dist = abs(fn - f1)
        curr_it = curr_it + 1

    if curr_it == maxit:
        raise ValueError("Max iterations; Convergence failed.")

    if ret_points:
        return x1, f1, curr_it, ret_points
    else:
        return x1, f1, curr_it

def plot_function_3d(func, area):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(
                       np.linspace(*area.x.to_tuple(), num=100),
                       np.linspace(*area.y.to_tuple(), num=100))
    Z = func([X, Y])
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
    cset = ax.contour(X, Y, Z, zdir='z', offset=0, stride=1, cmap=cm.coolwarm)

    ax.set_xlabel('X')
    ax.set_xlim(*area.x.to_tuple())
    ax.set_ylabel('Y')
    ax.set_ylim(*area.y.to_tuple())

    plt.show()

def plot_from_net(func, area, first_point):
    x = np.linspace(*area.x.to_tuple(), num=200)
    y = np.linspace(*area.y.to_tuple(), num=200)

    X, Y = np.meshgrid(x, y)
    Z = func([X, Y])

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection="3d")

    fig.suptitle("My Function", size=24)

    # Color mesh
    ax1.set_axis_bgcolor("white")
#    ax1.pcolormesh(X, Y, Z, cmap=cm.jet, norm=LogNorm())
    ax1.contour(X, Y, Z, zdir='z', offset=0, stride=0.1, cmap=cm.coolwarm)
    ax1.scatter(*first_point, color="k")
    ax1.annotate('First Point', xy=first_point, xytext=(-0.5, 1.25),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # Surface plot
    ax2.set_axis_bgcolor("white")
    ax2.plot_surface(X, Y, Z, norm = LogNorm(), cmap=cm.jet, linewidth=1)


    first_point_3d = (first_point[0], first_point[1], func(first_point))

    ax2.view_init(azim=65, elev=25)
    ax2.scatter(*first_point_3d, color="k")
    xa, ya, _ = proj3d.proj_transform(first_point_3d[0], first_point_3d[1], first_point_3d[2], ax2.get_proj())
    ax2.annotate("First Point", xy = (xa, ya), xytext = (-20, 30),
                 textcoords = 'offset points', ha = 'right', va = 'bottom',
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
#    plt.show()

    iterstosee = [0, 1, 2, 3, 5, 6, 7, 9, 10, 13, 17, 20, 25, 30, 50, 75, 90, 95]
    x, fx, its, ret_tris = nelder_mead(func, x0=np.array(first_point), tol=1e-12, iter_returns=iterstosee)
    print('Solution (%.6f, %.6f, %.6f) was found for %d iterations' % (x[0], x[1], fx, its))
    for i, tri in zip(iterstosee, ret_tris):
        print('Iteration #%d: [(%.3f,%.3f,%.3f),(%.3f,%.3f,%.3f),(%.3f,%.3f,%.2f)]' % (i,
              tri[0][0][0], tri[0][0][1], tri[0][1],
              tri[1][0][0], tri[1][0][1], tri[1][1],
              tri[2][0][0], tri[2][0][1], tri[2][1]))
    cols_n = 2
    rows_n = len(ret_tris) // 2
    if len(ret_tris) % 2 == 1:
        rows_n += 1

    fig, axs = plt.subplots(nrows=rows_n, ncols=cols_n, figsize=(8, 12))
    axs = axs.flatten()

    # Color mesh
    for i, curr_ax in enumerate(axs):
        if i == len(ret_tris):
            break
        verts = [ret_tris[i][j][0] for j in range(3)]
        curr_simplex = np.vstack(verts)
        #curr_ax.pcolormesh(X, Y, Z, cmap=cm.jet, norm=LogNorm())
        curr_ax.contour(X, Y, Z, zdir='z', offset=0, stride=0.1, cmap=cm.coolwarm)
        curr_ax.set_title("This is simplex for iteration %i" % iterstosee[i])
        curr_ax.scatter(curr_simplex[:, 0], curr_simplex[:, 1])
        print('curr ax is %r' % curr_ax)
        curr_ax.add_collection(PolyCollection([verts], facecolor=(1,1,0,0)))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    def FUNCTION(x):
        x1, x2 = x[0], x[1]
        return 6 * x1 ** 2 + 3 * x2 ** 2 - 4 * x1 * x2 + 4 * math.sqrt(5) * x1 + 8 * math.sqrt(5) * x2 + 22
    AREA = Area2d(Interval(-7, 4), Interval(-10, 2))
    FIRST_POINT = (-2, 1)
    #plot_function_3d(FUNCTION, AREA)
    plot_from_net(FUNCTION, AREA, FIRST_POINT)
