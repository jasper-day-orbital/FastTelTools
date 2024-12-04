from pyteltools.slf import Serafin
from pyteltools.slf.interpolation import MeshInterpolator
from fastteltools import PyMesh2D
import matplotlib.pyplot as plt
import numpy as np
import time
import os

DATA_PATH = "data/msh_hydro.slf"

with Serafin.Read(DATA_PATH, "fr") as resin:
    resin.read_header()
    header = resin.header.copy()
    # print(resin.header.summary())

    resin.get_time()
    # print(resin.time)

    t0 = time.perf_counter()
    # x, y = (
    #     resin.header.x[: resin.header.nb_nodes_2d].astype(np.float64),
    #     resin.header.y[: resin.header.nb_nodes_2d].astype(np.float64),
    # )
    # ikle = np.array(resin.header.ikle_2d) - 1
    # print(ikle.dtype)
    mesh = PyMesh2D(header)
    # oldmesh = MeshInterpolator(resin.header, True)
    # plt.scatter(oldmesh.points[:, 0], oldmesh.points[:, 1], c="k", s=0.5)
    # plt.scatter(x, y, s=0.5, c="b")
    # plt.show()

    t1 = time.perf_counter()
    print(f"Created mesh interpolator in {(t1 - t0)*1000} ms")

    # vs = resin.read_vars_in_frame(0, var_IDs=resin.header.var_IDs)
    values = resin.read_var_in_frame(0, "B")
    # U = resin.read_var_in_frame(0, "U")
    # V = resin.read_var_in_frame(0, "V")
    # vs is an 8x1035 ndarray
    # print(resin.header.var_IDs)
    # ['U', 'V', 'H', 'S', 'B', 'F', 'Q', 'M']

xm = mesh.points[0, :]
ym = mesh.points[1, :]
bounds = [np.min(xm), np.max(xm), np.min(ym), np.max(ym)]
xx = np.linspace(bounds[0], bounds[1], 200)
yy = np.linspace(bounds[2], bounds[3], 200)
points = np.array([[x, y] for x in xx for y in yy])

t0 = time.perf_counter()
coords, indices = mesh.get_point_interpolators(points)
t1 = time.perf_counter()

print(f"Interpolated points in {(t1-t0) * 1000} ms")

t0 = time.perf_counter()
_coords, _indices = mesh.get_point_interpolators_parallel(points)
t1 = time.perf_counter()

print(f"Interpolated points (parallel) in {(t1-t0) * 1000} ms")

print(
    sum([1 for coord in coords[:, 0] if not np.isnan(coord)]),
    "/",
    len(points),
    "points inside mesh",
)


t0 = time.perf_counter()

results = mesh.interpolate(coords, indices, values)
t1 = time.perf_counter()

print(f"Found values in {(t1 - t0) * 1000} ms")

# plt.imshow(np.array(results).reshape([200, 200]).T, extent=bounds, origin="lower")

# plt.scatter(mesh.points[0, :], mesh.points[1, :], c="r", s=1)
# # plt.quiver(xm, ym, U, V, color="white")
# plt.show()
