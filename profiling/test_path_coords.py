from fastteltools import PyPathCoords
import time


def main():
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.environ["RUST_BACKTRACE"] = "full"

    n_x = 500
    n_y = 500
    n_path = 100

    x = np.linspace(-10, 10, n_x)
    y = np.linspace(-10, 10, n_y)
    points = np.array([[_x, _y] for _x in x for _y in y])
    # path = np.array([[0.0, x] for x in np.linspace(-8, 8, 30)])
    path = np.array(
        [[4 * np.cos(th), 4 * np.sin(th)] for th in np.linspace(0, 2 * np.pi, n_path)]
    )

    t0 = time.perf_counter()
    inds, dists = PyPathCoords.get_continuous_path_coords(
        path.astype(np.float64), points.astype(np.float64)
    )
    t1 = time.perf_counter()
    print(f"{(t1 - t0) * 1e3} ms to complete operation")

    xx = points[:, 0].reshape(n_x, n_y)
    yy = points[:, 1].reshape(n_x, n_y)

    r = (xx**2 + yy**2) ** 0.5
    theta = np.atan2(yy, xx)
    theta[:, y < 0] += 2 * np.pi

    # Plotting
    plt.subplot(211)
    plt.scatter(path[:, 0], path[:, 1], s=0.5)
    plt.imshow(
        dists.reshape(n_x, n_y).T - np.abs(r - 4).T,
        origin="lower",
        extent=[-10, 10, -10, 10],
    )
    plt.colorbar()
    plt.title("Distances")
    plt.subplot(212)
    plt.scatter(path[:, 0], path[:, 1], s=0.5)
    plt.imshow(
        inds.reshape(n_x, n_y).T,
        origin="lower",
        extent=[-10, 10, -10, 10],
    )
    plt.colorbar()
    plt.title("Indices")
    plt.show()


if __name__ == "__main__":
    main()
