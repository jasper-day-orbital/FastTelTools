use approx::abs_diff_eq;
use ndarray::prelude::*;
use num::{traits::float::TotalOrder, Float};
use numpy::{PyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{prelude::*, pyclass};

/// Finds the distances from path to points
/// Takes:
/// path: A nx2 list of (x, y) coordinates defining the path
/// points: An mx2 list of (x, y) coordinates defining the points
/// Returns:
/// indices: A mx1 vector indexing the path-point closest to each point
/// distances: the distance from each point to the nearest point on the path
fn get_path_coords<T: Float + TotalOrder + 'static>(
    path: &Array2<T>,
    points: &Array2<T>,
) -> (Array1<usize>, Array1<T>) {
    let result: Vec<(usize, T)> = points
        .rows()
        .into_iter()
        .map(|point| {
            // Get path distances
            let squared_distances = (path.clone() - point).pow2().sum_axis(Axis(1));
            squared_distances
                .into_iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
        })
        .collect();

    (
        // indices
        Array::from_iter(result.iter().map(|x| x.0)),
        // distances
        Array::from_iter(result.iter().map(|x| x.1.sqrt())),
    )
}

fn minimize_distance_quadratic(
    p: &ArrayView1<f64>,
    xleft: &ArrayView1<f64>,
    x: &ArrayView1<f64>,
    xright: &ArrayView1<f64>,
) -> (f64, f64) {
    fn find_coefficients(left: f64, center: f64, right: f64) -> (f64, f64, f64) {
        // Get the coefficients of the quadratic approximation through left, center, right
        let c3 = center;
        let v1 = right - center;
        let v2 = left - center;
        let c1 = (v1 + v2) / 2.0;
        let c2 = (v1 - v2) / 2.0;
        (c1, c2, c3)
    }
    let x_c = find_coefficients(xleft[0], x[0], xright[0]);
    let y_c = find_coefficients(xleft[1], x[1], xright[1]);
    let gx = |xi: f64| x_c.0 * xi.powi(2) + x_c.1 * xi + x_c.2;
    let gy = |xi: f64| y_c.0 * xi.powi(2) + y_c.1 * xi + y_c.2;
    let gpx = |xi| 2.0 * x_c.0 * xi + x_c.1;
    let gpy = |xi| 2.0 * y_c.0 * xi + y_c.1;
    let gppx = |_xi: f64| 2.0 * x_c.0;
    let gppy = |_xi: f64| 2.0 * y_c.0;
    let d2 = |xi: f64| (p[0] - gx(xi)).powi(2) + (p[1] - gy(xi)).powi(2);
    let d2p = |xi: f64| -2.0 * ((p[0] - gx(xi)) * gpx(xi) + (p[1] - gy(xi)) * gpy(xi));
    let d2pp = |xi: f64| {
        -2.0 * ((p[0] - gx(xi)) * gppx(xi) - (p[0] - gx(xi)) * gpx(xi) * gpx(xi)
            + (p[1] - gy(xi)) * gppy(xi)
            - (p[0] - gy(xi)) * gpy(xi) * gpy(xi))
    };

    // // find root of d2p (minimizing dp)
    // let mut xi = 0.0;
    // for _ in 0..10 {
    //     // 10 iterations of newton raphson
    //     let y = d2p(xi);
    //     let dy = d2pp(xi);
    //     xi -= y / dy;
    //     xi = xi.clamp(-1.0, 1.0);
    // } // quadratic convergence squares the error at each step

    let mut a = -1.0;
    let mut b = 1.0;
    let mut xi = 0.0;

    // 10 iterations of bisection method
    for _ in 0..20 {
        if d2p(xi).signum() == d2p(a).signum() {
            a = xi;
        } else {
            b = xi
        }
        xi = (a + b) / 2.0;
    }

    // Calculate the distance to the quadratic approximation
    (xi, d2(xi).powf(0.5))
}

fn minimize_distance_linear(
    p: &ArrayView1<f64>,
    xleft: &ArrayView1<f64>,
    xright: &ArrayView1<f64>,
) -> (f64, f64) {
    let distance = |a: &ArrayView1<f64>, b: &ArrayView1<f64>| (a - b).pow2().sum().sqrt();
    let l2 = distance(xleft, xright);
    if abs_diff_eq!(l2, 0.0) {
        (0.0, distance(xleft, p))
    } else {
        let t = (p - xleft).dot(&(xright - xleft)) / l2;
        let t_clamp = t.clamp(0.0, 1.0);
        let projection = xleft + t_clamp * (xright - xleft);
        (t_clamp, distance(p, &projection.view()))
    }
}

fn get_continuous_path_coords(
    path: &Array2<f64>,
    points: &Array2<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let (indices, _) = get_path_coords(path, points);
    let n_path = path.shape()[0];
    let result: Vec<(f64, f64)> = points
        .rows()
        .into_iter()
        .zip(indices.iter())
        .map(|(point, &index)| {
            if index == 0 {
                minimize_distance_linear(&point, &path.slice(s![0, ..]), &path.slice(s![1, ..]))
            } else if index == n_path - 1 {
                minimize_distance_linear(
                    &point,
                    &path.slice(s![index - 1, ..]),
                    &path.slice(s![index, ..]),
                )
            } else {
                // assert!(index < n_path);
                minimize_distance_quadratic(
                    &point,
                    &path.slice(s![index - 1, ..]),
                    &path.slice(s![index, ..]),
                    &path.slice(s![index + 1, ..]),
                )
                // let left = minimize_distance_linear(
                //     &point,
                //     &path.slice(s![index - 1, ..]),
                //     &path.slice(s![index, ..]),
                // );
                // let right = minimize_distance_linear(
                //     &point,
                //     &path.slice(s![index, ..]),
                //     &path.slice(s![index + 1, ..]),
                // );
                // if right.1 > left.1 {
                //     left
                // } else {
                //     right
                // }
            }
        })
        .collect();
    (
        Array::from_iter(
            result
                .iter()
                .zip(indices.iter())
                .map(|(x, ind)| x.0 + *ind as f64),
        ),
        Array::from_iter(result.iter().map(|x| x.1)),
    )
}

type Result<'py, T, U> = (Bound<'py, PyArray1<T>>, Bound<'py, PyArray1<U>>);

#[pyclass(subclass)]
pub struct _PyPathCoords {}

#[pymethods]
impl _PyPathCoords {
    #[staticmethod]
    fn get_path_coords<'py>(
        py: Python<'py>,
        path: PyReadonlyArray2<f64>,
        points: PyReadonlyArray2<f64>,
    ) -> Result<'py, usize, f64> {
        let path = path.as_array().to_owned();
        let points = points.as_array().to_owned();
        let result = get_path_coords(&path, &points);
        (result.0.to_pyarray_bound(py), result.1.to_pyarray_bound(py))
    }
    #[staticmethod]
    fn get_continuous_path_coords<'py>(
        py: Python<'py>,
        path: PyReadonlyArray2<f64>,
        points: PyReadonlyArray2<f64>,
    ) -> Result<'py, f64, f64> {
        let path = path.as_array().to_owned();
        let points = points.as_array().to_owned();
        let result = get_continuous_path_coords(&path, &points);
        (result.0.to_pyarray_bound(py), result.1.to_pyarray_bound(py))
    }
}
