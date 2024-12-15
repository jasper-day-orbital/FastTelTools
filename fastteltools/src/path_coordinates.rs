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
    path: Array2<T>,
    points: Array2<T>,
) -> (Array1<usize>, Array1<T>) {
    let n_points = points.shape()[0];
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

    let mut indices = Array1::<usize>::zeros(n_points);
    let mut distances = Array1::<T>::zeros(n_points);

    indices
        .iter_mut()
        .zip(distances.iter_mut())
        .zip(result.into_iter())
        .for_each(|((index, distance), (path_index, path_distance))| {
            *index = path_index;
            *distance = path_distance.sqrt();
        });
    (indices, distances)
}

type Result<'py, T> = (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<T>>);

#[pyclass(subclass)]
pub struct _PyPathCoords {}

#[pymethods]
impl _PyPathCoords {
    #[staticmethod]
    fn get_path_coords<'py>(
        py: Python<'py>,
        path: PyReadonlyArray2<f64>,
        points: PyReadonlyArray2<f64>,
    ) -> Result<'py, f64> {
        let path = path.as_array().to_owned();
        let points = points.as_array().to_owned();
        let result = get_path_coords(path, points);
        (result.0.to_pyarray_bound(py), result.1.to_pyarray_bound(py))
    }
}
