from pyteltools.slf import Serafin
from pyteltools.slf.interpolation import MeshInterpolator
import matplotlib.pyplot as plt
import numpy as np
import time
import os

DATA_PATH = "data/msh_hydro.slf"

with Serafin.Read(DATA_PATH, "fr") as resin:
    resin.read_header()
    # print(resin.header.summary())

    resin.get_time()
    # print(resin.time)

    t0 = time.perf_counter()
    mesh = MeshInterpolator(resin.header, True)
    t1 = time.perf_counter()
    print(f"Created mesh interpolator in {(t1 - t0)*1000} ms")

    # vs = resin.read_vars_in_frame(0, var_IDs=resin.header.var_IDs)
    values = resin.read_var_in_frame(0, "B")
    # vs is an 8x1035 ndarray
    # print(resin.header.var_IDs)
    # ['U', 'V', 'H', 'S', 'B', 'F', 'Q', 'M']

xm = mesh.points[:, 0]
ym = mesh.points[:, 1]
bounds = [np.min(xm), np.max(xm), np.min(ym), np.max(ym)]
xx = np.linspace(bounds[0], bounds[1], 200)
yy = np.linspace(bounds[2], bounds[3], 200)
points = [[x, y] for x in xx for y in yy]

t0 = time.perf_counter()
is_inside, point_interpolators = mesh.get_point_interpolators(points=points)
t1 = time.perf_counter()

print(f"Interpolated points in {(t1-t0) * 1000} ms")

print(sum(map(int, is_inside)), "/", len(points), "points inside mesh")


t0 = time.perf_counter()
results = []
for point_interpolator in point_interpolators:
    if point_interpolator is not None:
        (i, j, k), interpolator = point_interpolator
        results.append(interpolator.dot(values[[i, j, k]]))
    else:
        results.append(np.nan)
t1 = time.perf_counter()

print(f"Found values in {(t1 - t0) * 1000} ms")

plt.imshow(np.array(results).reshape([200, 200]).T, extent=bounds)
plt.show()
