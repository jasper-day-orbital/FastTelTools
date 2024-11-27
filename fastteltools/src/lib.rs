use ndarray::{arr1, s, Array1, Array2};
use numpy::{PyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

fn barycentric_coordinates(t: Array2<f64>, x: f64, y: f64) -> Array1<f64> {
    // T is a 2d array where each row is a vertex of the triangle
    let x1 = t[(0, 0)];
    let y1 = t[(0, 1)];
    let x2 = t[(1, 0)];
    let y2 = t[(1, 1)];
    let x3 = t[(2, 0)];
    let y3 = t[(2, 1)];
    let vec_x = arr1(&[x2 - x3, x3 - x1, x1 - x2]);
    let vec_y = arr1(&[y2 - y3, y3 - y1, y1 - y2]);
    let norm_z = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
    let vec_norm_z = arr1(&[norm_z, 0., 0.]);
    (vec_norm_z + (x - x1) * vec_y - (y - y1) * vec_x) / norm_z
}

#[pyfunction]
fn interpolate<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<f64>,
    x: f64,
    y: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = t.as_array();
    PyResult::Ok(
        barycentric_coordinates(t.to_owned(), x, y)
            .to_pyarray_bound(py)
            .to_owned(),
    )
}

/// A Python module implemented in Rust.
#[pymodule]
fn fastteltools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interpolate, m)?)?;
    Ok(())
}
