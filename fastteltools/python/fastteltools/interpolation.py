from ._fastteltools import _PyMesh2D
import numpy as np


class PyMesh2D(_PyMesh2D):
    "Fast interpolation on unstructured Serafin meshes"

    def __new__(cls, header):
        x = header.x[: header.nb_nodes_2d]
        y = header.y[: header.nb_nodes_2d]
        # Back to a zero-based indexing system
        ikle = np.array(header.ikle_2d - 1)
        return super().__new__(cls, x, y, ikle)

    def interpolate_points(self, points, values):
        coords, inds = self.get_point_interpolators(points)
        return np.sum(coords * values[inds], axis=1)

    @staticmethod
    def interpolate(self, coords, inds, values):
        return np.sum(coords * values[inds], axis=1)
