use ndarray::{arr1, arr2, s, stack, Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{prelude::*, pyclass};
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};

#[derive(Clone, Debug)]
struct Triangle {
    coords: Array2<f64>,
    indices: [usize; 3],
    corners: [[f64; 2]; 2],
}

fn barycentric_coordinates(t: &Array2<f64>, x: f64, y: f64) -> Array1<f64> {
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

#[inline]
fn is_inside(coords: &Array1<f64>) -> bool {
    coords.iter().all(|&x| x >= 0. && x <= 1.)
}

fn get_corners(coords: &Array2<f64>) -> [[f64; 2]; 2] {
    let mut c1 = [0.; 2];
    let mut c2 = [0.; 2];

    for (i, col) in coords.axis_iter(Axis(1)).enumerate() {
        let (min, max) = col
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| {
                (min.min(*v), max.max(*v))
            });
        // let min = *col.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        // let max = *col.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
        c1[i] = min;
        c2[i] = max;
    }
    [c1, c2]
}

impl Triangle {
    fn new(coords: Array2<f64>, indices: [usize; 3]) -> Triangle {
        Triangle {
            corners: get_corners(&coords),
            coords: coords,
            indices: indices,
        }
    }

    fn get_corners(&self) -> [[f64; 2]; 2] {
        self.corners
    }

    fn barycentric_coordinates(&self, point: &[f64; 2]) -> Array1<f64> {
        barycentric_coordinates(&self.coords, point[0], point[1])
    }
}

impl RTreeObject for Triangle {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        let corners = self.get_corners();
        AABB::from_corners(corners[0], corners[1])
    }
}

impl PointDistance for Triangle {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        self.envelope().distance_2(point)
    }

    fn contains_point(&self, point: &<Self::Envelope as rstar::Envelope>::Point) -> bool {
        is_inside(&self.barycentric_coordinates(point))
    }
}

fn make_index(x: Array1<f64>, y: Array1<f64>, ikle: Vec<[usize; 3]>) -> RTree<Triangle> {
    let elements = ikle
        .iter()
        .map(|edges| {
            Triangle::new(
                arr2(&[
                    [x[edges[0]], y[edges[0]]],
                    [x[edges[1]], y[edges[1]]],
                    [x[edges[2]], y[edges[2]]],
                ]),
                edges.clone(),
            )
        })
        .collect();
    RTree::bulk_load(elements)
}

#[derive(Debug)]
struct Mesh2D {
    points: Array2<f64>,
    rtree: RTree<Triangle>,
}

struct CoordResult {
    indices: [usize; 3],
    coords: Array1<f64>,
}

impl Mesh2D {
    fn new(x: Array1<f64>, y: Array1<f64>, ikle: Vec<[usize; 3]>) -> Mesh2D {
        Mesh2D {
            points: stack![Axis(0), x, y],
            rtree: make_index(x, y, ikle),
        }
    }

    fn locate_at_point(&self, point: &[f64; 2]) -> Option<&Triangle> {
        self.rtree.locate_at_point(point)
    }

    fn locate_in_envelope_intersecting(
        &self,
        envelope: &<Triangle as RTreeObject>::Envelope,
    ) -> Vec<&Triangle> {
        self.rtree
            .locate_in_envelope_intersecting(envelope)
            .collect()
    }

    fn get_point_interpolators(&self, points: Vec<[f64; 2]>) -> Vec<Option<CoordResult>> {
        points
            .iter()
            .map(|point| match self.locate_at_point(point) {
                None => None,
                Some(tri) => Some(CoordResult {
                    indices: tri.indices,
                    coords: tri.barycentric_coordinates(point),
                }),
            })
            .collect()
    }

    fn get_point_interpolators_parallel(&self, points: Vec<[f64; 2]>) -> Vec<Option<CoordResult>> {
        points
            // do it in parallel
            .par_iter()
            .map(|point| match self.locate_at_point(point) {
                None => None,
                Some(tri) => Some(CoordResult {
                    indices: tri.indices,
                    coords: tri.barycentric_coordinates(point),
                }),
            })
            .collect()
    }

    // fn get_point_interpolators_envelope(&self, points: Vec<[f64; 2]>) -> Vec<Option<CoordResult>> {
    //     points
    //         .iter()
    //         .map(|point| match self.locate_in_envelope_intersecting(point) {
    //             None => None,
    //             Some(tri) => Some(CoordResult {
    //                 indices: tri.indices,
    //                 coords: tri.barycentric_coordinates(point),
    //             }),
    //         })
    //         .collect()
    // }
}

#[pyclass(subclass)]
struct PyMesh2D {
    index: Mesh2D,
}

#[pyclass]
struct PyCoordResult {
    indices: [usize; 3],
    coords: PyObject,
}

#[pymethods]
impl PyCoordResult {
    #[getter]
    fn get_indices(&self) -> [usize; 3] {
        self.indices
    }

    #[getter]
    fn get_coords(&self) -> &PyObject {
        &self.coords
    }
}

impl CoordResult {
    fn to_pycoordresult<'py>(&self, py: Python<'py>) -> PyCoordResult {
        PyCoordResult {
            indices: self.indices,
            coords: self.coords.to_pyarray_bound(py).to_object(py),
        }
    }
}

#[pymethods]
impl PyMesh2D {
    #[new]
    fn new(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>, ikle: Vec<[usize; 3]>) -> PyMesh2D {
        let x = x.as_array().to_owned();
        let y = y.as_array().to_owned();
        PyMesh2D {
            index: Mesh2D::new(x, y, ikle),
        }
    }

    #[getter]
    fn points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.index.points.to_pyarray_bound(py)
    }

    // fn get_point_interpolators_envelope<'py>(
    //     &self,
    //     py: Python<'py>,
    //     points: Vec<PyReadonlyArray1<f64>>,
    // ) -> Vec<Option<PyCoordResult>> {
    //     let points = points
    //         .into_iter()
    //         .map(|point| point.as_array().to_owned())
    //         .map(|arr| [arr[0], arr[1]])
    //         .collect();
    //     let interpolators = self.index.get_point_interpolators_envelope(points);
    //     interpolators
    //         .into_iter()
    //         .map(|interpolator| match interpolator {
    //             None => None,
    //             Some(coord) => Some(coord.to_pycoordresult(py)),
    //         })
    //         .collect()
    // }

    fn get_point_interpolators<'py>(
        &self,
        py: Python<'py>,
        points: Vec<PyReadonlyArray1<f64>>,
    ) -> Vec<Option<PyCoordResult>> {
        let points = points
            .into_iter()
            .map(|point| point.as_array().to_owned())
            .map(|arr| [arr[0], arr[1]])
            .collect();
        let interpolators = self.index.get_point_interpolators(points);
        interpolators
            .into_iter()
            .map(|interpolator| match interpolator {
                None => None,
                Some(coord) => Some(coord.to_pycoordresult(py)),
            })
            .collect()
    }

    /// Optimized parallel point interpolators
    fn get_point_interpolators_parallel<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<f64>,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        // Convert input points to a native Rust structure
        let points = points.as_array();
        let num_points = points.len_of(Axis(0));

        // Prepare a buffer for indices and barycentric coordinates
        let mut results: Vec<Option<(Array1<f64>, [usize; 3])>> = vec![None; num_points];

        // Parallel processing of points
        results.par_iter_mut().enumerate().for_each(|(i, res)| {
            let point = [points[[i, 0]], points[[i, 1]]];
            *res = self
                .index
                .locate_at_point(&point)
                .map(|tri| (tri.barycentric_coordinates(&point), tri.indices));
        });

        // Flatten the results into a NumPy array
        let mut output = Array2::<f64>::zeros((num_points, 6)); // 6: 3 barycentric coords + 3 indices

        for (i, result) in results.into_iter().enumerate() {
            if let Some((coords, indices)) = result {
                // Fill barycentric coordinates
                for j in 0..3 {
                    output[[i, j]] = coords[j];
                }
                // Fill triangle indices
                for j in 0..3 {
                    output[[i, 3 + j]] = indices[j] as f64;
                }
            } else {
                // Mark as invalid (optional, based on your application)
                output.row_mut(i).slice_mut(s![..3]).fill(f64::NAN);
                output.row_mut(i).slice_mut(s![3..]).fill(-1f64);
            }
        }

        // Return the NumPy array back to Python
        Some(output.to_pyarray_bound(py))
    }
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
        barycentric_coordinates(&t.to_owned(), x, y)
            .to_pyarray_bound(py)
            .to_owned(),
    )
}

/// A Python module implemented in Rust.
#[pymodule]
fn fastteltools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interpolate, m)?)?;
    m.add_class::<PyMesh2D>()?;
    m.add_class::<PyCoordResult>()?;
    Ok(())
}
