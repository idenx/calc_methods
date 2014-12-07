import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
import math
from operator import attrgetter

DEBUG = False

def debug(s):
    if DEBUG:
        print(s)

class OptimizationResult(object):
    def __init__(self):
        self._points = []
        self.solution = None
        self.iterations_n = 0
    def add_point(self, point):
        self._points.append(point)
    def add_points(self, *points):
        self._points.extend(points)
    def register_iteration(self):
        self.iterations_n += 1
    @property
    def points(self):
        return np.array(self._points)

def is_number_inside_interval(n, interval):
    return n > interval[0] and n < interval[1]

def find_minimum_by_library(func, interval, tol):
    ores = minimize(func, [interval[0]], bounds=[interval], method='SLSQP', tol=tol)
    r = OptimizationResult()
    r.solution = (ores.x, func(ores.x))
    debug('library res is %r' % ores)
    r.iterations_n = ores.get('nit', 0)
    return r

def find_minimum_by_bitwise_search(func, interval, tol):
    x0, y0 = interval[0], func(interval[0])
    k = 4.0
    step = (interval[1] - interval[0]) / k
    res = OptimizationResult()
    res.add_point((x0, y0))

    debug('starting with (x, y, step) = (%.7f, %.7f, %.7f)' % (x0, y0, step))
    while abs(step) > tol:
        res.register_iteration()
        x1, y1 = x0 + step, func(x0 + step)
        debug('iteration #%d: (x, y, step) = (%.7f, %.7f, %.7f)' % (res.iterations_n, x1, y1, step))
        res.add_point((x1, y1))
        need_to_change_direction = y1 > y0 or not is_number_inside_interval(x1, interval)
        x0, y0 = x1, y1
        if need_to_change_direction:
            step /= -k
            debug('changing direction, new_step is %.7f' % step)
            continue
    res.solution = (x0, y0)
    return res

GOLDEN_SECTION_COEF = 1.0 - (2.0 / (1 + math.sqrt(5))) # 0.382...

def find_minimum_by_golden_section_search(f, interval, tol):
    a, b = interval
    k = GOLDEN_SECTION_COEF
    delta = k * (b - a)
    x1, x2 = a + delta, b - delta
    A, B = f(x1), f(x2)

    res = OptimizationResult()
    res.add_point((x1, A))
    res.add_point((x2, B))

    while True:
        res.register_iteration()

        a, b = (a, x2) if A < B else (x1, b)
        int_len = b - a
        if int_len <= tol:
            break

        delta = k * int_len
        if A < B:
            x2, B = x1, A
            x1 = a + delta
            A = f(x1)
            res.add_point((x1, A))
        else:
            x1, A = x2, B
            x2 = b - delta
            B = f(x2)
            res.add_point((x2, B))

    x = (a + b) / 2.0
    res.solution = (x, f(x))
    return res

def find_parabola_vertex_from_3_points(points):
    (x1, y1), (x2, y2), (x3, y3) = [p.to_tuple() for p in points]
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A     = float(x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B     = float(x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom
    return -B / (2 * A)

def find_parabola_vertex_v2(x1, x2, x3, y1, y2, y3):
    return 0.5 * float((x2**2 - x3**2) * y1 + (x3**2 - x1**2) * y2 + (x1**2 - x2**2) * y3) / ((x2 - x3) * y1 + (x3 - x1) * y2 + (x1 - x2) * y3)

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_tuple(self):
        return (self.x, self.y)

def find_minimum_by_quadratic_approximation(f, interval, tol):
    x1, x3 = interval
    x2 = x1 + GOLDEN_SECTION_COEF * (x3 - x1)
    y1, y2, y3 = f(x1), f(x2), f(x3)
    p1, p2, p3 = Point(x1, y1), Point(x2, y2), Point(x3, y3)
    res = OptimizationResult()
    res.add_points(p1.to_tuple(), p3.to_tuple(), p2.to_tuple())
    points = [p1, p2, p3]

    while True:
        res.register_iteration()
        points.sort(key=attrgetter('y'))
        v = find_parabola_vertex_from_3_points(points)
        vertex = Point(v, f(v))

        min_point = points[0]
        res.add_point(min_point.to_tuple())
        if abs(vertex.x - min_point.x) <= tol:
            x = (vertex.x + min_point.x) / 2.0
            res.solution = (x, f(x))
            return res
        points[2] = vertex # replace point with max y with vertex

def find_first_derivative(f, x, delta):
    return 0.5 * float(f(x + delta) - f(x - delta)) / delta

def find_second_derivative(f, x, delta):
    r = float(f(x + delta) - 2 * f(x) + f(x - delta)) / (delta * delta)
    assert r, '2nd derivative is %.10f, f(x+delta)=%f,f(x)=%f,f(x-delta)=%f,delta=%.10f,delta*delta=%.10f' % (r, f(x + delta), f(x), f(x-delta), delta, delta*delta)
    return r

def find_minimum_by_newton_method(f, interval, tol):
    x0, x1 = interval
    x1 = x0 + GOLDEN_SECTION_COEF * (x1 - x0)
    res = OptimizationResult()
    delta = tol / 10.0

    while True:
        res.register_iteration()
        res.add_point((x1, f(x1)))
        x0 = x1
        x1 = x0 - find_first_derivative(f, x0, delta) / find_second_derivative(f, x0, delta)
        if abs(x1 - x0) <= tol or x1 < interval[0] or x1 > interval[1]:
            res.solution = (x1, f(x1))
            return res

def get_func_points(f, interval, points_n=100):
    x = np.linspace(interval[0], interval[1], num=points_n)
    return (x, np.vectorize(f)(x))

class Plot(object):
    def draw_line(self, x, y, *args, **kwargs):
        plt.plot(x, y, *args, **kwargs)
    def draw_arrows(self, x, y, *args, **kwargs):
        plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], *args,
                   scale_units='xy', angles='xy', scale=1, width=0.005, headwidth=3.0, **kwargs)
    def draw_points(self, points, *args, **kwargs):
        for p in points:
            self.draw_line(p[0], p[1], *args, **kwargs)
    def show(self):
        plt.show()

def visualize_solution(r, draw_type=None):
    plot = Plot()
    plot.draw_line(*get_func_points(FUNC, INTERVAL), color='green')
    plot.draw_line(*r.solution, color='red', markersize=10.0, marker='+')
    if r.points.size:
        if draw_type == 'arrows':
            plot.draw_arrows(r.points[:,0], r.points[:,1], color='blue')
        elif draw_type == 'points':
            plot.draw_points(r.points, color='black', markersize=10.0, marker='+')
    plot.show()

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

if __name__ == '__main__':
    INTERVAL = [1,2]
    FUNC = lambda x: np.arctan(x ** 3 - 5 * x + 1) + (x * x / (3 * x - 2)) ** np.sqrt(3)

    tolerances = (0.01, 0.0001, 0.000001)
    methods = {
        'bitwise': find_minimum_by_bitwise_search,
        'golden_section': find_minimum_by_golden_section_search,
        'quadratic_approximation': find_minimum_by_quadratic_approximation,
        'newton': find_minimum_by_newton_method,
        'library': find_minimum_by_library
    }

    print('-----')
    results = []
    for tol in tolerances:
        res = {}
        for method, optimizer in methods.iteritems():
            f = CountingFunction(FUNC)
            r = optimizer(f, INTERVAL, tol=tol)
            print('%s: x_min = %.7f, y_min = %.7f, tol=%.7f, iter_n=%d, fev_n=%d' % (method, r.solution[0], r.solution[1],
                  tol, r.iterations_n, f.calls_count))
            res[method] = r
        print('-----')
        results.append(res)

    #visualize_solution(results[0]['bitwise'], draw_type='arrows')
    #visualize_solution(results[0]['golden_section'], draw_type='points')
    #visualize_solution(results[0]['quadratic_approximation'], draw_type='points')
    visualize_solution(results[0]['newton'], draw_type='points')

