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
from scipy.optimize import minimize

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
    init_guess = x0
    fx0 = f(x0)
    dist = float('inf')
    curr_it = 0

    n = np.size(x0) # dimensions count

    alpha = 1.0
    beta = 1.0 + (2.0 / n)
    gamma = 0.75 - 1.0 / (2.0 * n)
    delta = 1.0 - (1.0 / n)

    # Create the simplex points and do the initial sort
    simplex_points = np.empty((n+1, n))

    pt_fval = [(x0, fx0)]

    simplex_points[0, :] = x0

    for ind, elem in enumerate(x0):
        curr_tau = 0.05
        curr_point = np.squeeze(np.eye(1, M=n, k=ind)*curr_tau + x0)
        simplex_points[ind, :] = curr_point
        pt_fval.append((curr_point, f(curr_point)))

    if iter_returns is not None:
        ret_points = []
    else:
        ret_points = None


    # The Core of The Nelder-Mead Algorithm
    while dist>tol and curr_it<maxit:

        # 1: Sort and find new center point (excluding worst point)
        pt_fval = sorted(pt_fval, key=lambda v: v[1])
        xbar = x0 * 0

        for i in range(n):
            xbar = xbar + (pt_fval[i][0])/(n)

        if iter_returns is not None and curr_it in iter_returns:
            ret_points.append(pt_fval)

        # Define useful variables
        x1, f1 = pt_fval[0] # lowest point
        xn, fn = pt_fval[n-1] # point between lowest and highest
        xnp1, fnp1 = pt_fval[n] # highest (worst) point

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

def plot_function_and_solution(func, area, first_point, solution, meshgrid):
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection="3d")

    fig.suptitle("My Function", size=24)

    # Color mesh
    ax1.set_axis_bgcolor("white")
#    ax1.pcolormesh(X, Y, Z, cmap=cm.jet, norm=LogNorm())
    ax1.contour(*meshgrid, zdir='z', offset=0, stride=0.1, cmap=cm.coolwarm)
    ax1.scatter(*first_point, color="k")
    ax1.annotate('First Point', xy=first_point, xytext=(-0.5, 1.25),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    ax1.scatter(solution[0], solution[1], color="g")
    ax1.annotate('Solution', xy=(solution[0], solution[1]), xytext=(-1, -3),
                 arrowprops=dict(facecolor='blue', shrink=0.05))

    # Surface plot
    ax2.set_axis_bgcolor("white")
    ax2.plot_surface(*meshgrid, norm = LogNorm(), cmap=cm.jet, linewidth=1)

    first_point_3d = (first_point[0], first_point[1], func(first_point))

    ax2.view_init(azim=65, elev=25)
    ax2.scatter(*first_point_3d, color="k")
    xa, ya, _ = proj3d.proj_transform(first_point_3d[0], first_point_3d[1], first_point_3d[2], ax2.get_proj())
    ax2.annotate("First Point", xy = (xa, ya), xytext = (20, 120),
                 textcoords = 'offset points', ha = 'right', va = 'bottom',
                 arrowprops=dict(facecolor='black', shrink=0.05))
    ax2.scatter(*solution, color="k")
    xa, ya, _ = proj3d.proj_transform(solution[0], solution[1], solution[2], ax2.get_proj())
    ax2.annotate("Solution", xy = (xa, ya), xytext = (0, 100),
                 textcoords = 'offset points', ha = 'right', va = 'bottom',
                 arrowprops=dict(facecolor='blue', shrink=0.05))

def plot_iterations(iters, meshgrid):
    for i, tri in iters:
        print('Iteration #%d: [(%.3f,%.3f,%.3f),(%.3f,%.3f,%.3f),(%.3f,%.3f,%.2f)]' % (i,
              tri[0][0][0], tri[0][0][1], tri[0][1],
              tri[1][0][0], tri[1][0][1], tri[1][1],
              tri[2][0][0], tri[2][0][1], tri[2][1]))
    cols_n = 3
    rows_n = int(math.ceil(len(iters) / cols_n))

    fig, axs = plt.subplots(nrows=rows_n, ncols=cols_n)
    axs = axs.flatten()

    # Color mesh
    for i, curr_ax in enumerate(axs):
        if i == len(iters):
            break
        verts = [iters[i][1][j][0] for j in range(3)]
        curr_simplex = np.vstack(verts)
        #curr_ax.pcolormesh(X, Y, Z, cmap=cm.jet, norm=LogNorm())
        curr_ax.contour(*meshgrid, zdir='z', offset=0, stride=0.1, cmap=cm.coolwarm)
        curr_ax.set_title("Iteration #%d" % iters[i][0])
        curr_ax.scatter(curr_simplex[:, 0], curr_simplex[:, 1])
        curr_ax.add_collection(PolyCollection([verts], facecolor=(1,1,0,0)))

    plt.tight_layout()
    plt.show()

def get_function_meshgrid(func, area, num=200):
    x = np.linspace(*area.x.to_tuple(), num=num)
    y = np.linspace(*area.y.to_tuple(), num=num)

    X, Y = np.meshgrid(x, y)
    Z = func([X, Y])
    return (X, Y, Z)

class CountingFunction(object):
    def __init__(self, func):
        self._calls_n = 0
        self._func = func
    def __call__(self, *args, **kwargs):
        self._calls_n += 1
        return self._func(*args, **kwargs)
    @property
    def calls_count(self):
        return self._calls_n
    def reset_counter(self):
        self._calls_n = 0


def plot_function_3d(func, area, num=100):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(
                       np.linspace(*area.x.to_tuple(), num=num),
                       np.linspace(*area.y.to_tuple(), num=num))
    Z = func([X, Y])
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
#    cset = ax.contour(X, Y, Z, zdir='z', offset=0, stride=0.5, cmap=cm.coolwarm)

    ax.set_xlabel('X')
    ax.set_xlim(*area.x.to_tuple())
    ax.set_ylabel('Y')
    ax.set_ylim(*area.y.to_tuple())

    plt.show()

def minimize_function(func, area, first_point, tol, iters, draw_func=False, draw_func_and_solution=True, draw_iters=True):
    meshgrid = get_function_meshgrid(func, area, num=300)
    func = CountingFunction(func)
    x, fx, its, ret_tris = nelder_mead(func, x0=np.array(first_point), tol=tol, iter_returns=iters)
    solution = (x[0], x[1], fx)
    print('Solution by our impl (%.3f, %.3f, %.3f) was found for %d iterations and %d func evaluations' % (solution[0], solution[1], solution[2], its, func.calls_count))

    # calc by library
    func.reset_counter()
    res = minimize(func, x0=np.array(first_point), method='Nelder-Mead')
    print('Solution by library function %.3f, %.3f, %.3f) was found for %d iterations and %d func evaluations' % (res.x[0], res.x[1], res.fun, res.nit, func.calls_count))

    if draw_func:
        plot_function_3d(func, area, num=100)
    if draw_func_and_solution:
        plot_function_and_solution(func, area, first_point, solution, meshgrid)
    if draw_iters:
        plot_iterations(list(zip(iters, ret_tris)), meshgrid)

if __name__ == '__main__':
    def FUNCTION(x):
        x1, x2 = x[0], x[1]
        return 6 * x1 ** 2 + 3 * x2 ** 2 - 4 * x1 * x2 + 4 * math.sqrt(5) * x1 + 8 * math.sqrt(5) * x2 + 22
    def FUNCTION2(x):
        x1, x2 = x[0], x[1]
        return x1 ** 2 + x2 + 2 / (x1 ** 2 * x2 ** 2 + 0.1) - 3 * np.arctan(2 * x1) + 3 * x2
        #return x1 ** 2 + x2 + 2 / (x1 * x2) - 3 * np.arctan(2 * x1) + 3 * x2

    func = CountingFunction(FUNCTION)
    AREA = Area2d(Interval(-7, 4), Interval(-10, 2))
    AREA2 = Area2d(Interval(-2, 6), Interval(-2, 6))
    FIRST_POINT = (-2, 1)
    FIRST_POINT2 = (3, 3)
    TOLERANCE = 1e-6
    TOLERANCE2 = 1e-4
    ITERS = [0, 1, 5, 10, 13, 14, 15, 16, 17, 20, 30, 50]

    minimize_function(FUNCTION, AREA, FIRST_POINT, TOLERANCE, ITERS)
    minimize_function(FUNCTION2, AREA2, FIRST_POINT2, TOLERANCE2, ITERS, draw_func=True)
