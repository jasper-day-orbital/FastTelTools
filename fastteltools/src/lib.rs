use ndarray::{arr1, arr2, stack, Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{prelude::*, pyclass};
use rstar::{PointDistance, RTree, RTreeObject, AABB};

#[derive(Clone, Debug)]
struct Triangle {
    coords: Array2<f64>,
    indices: [usize; 3],
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

fn is_inside(coords: &Array1<f64>) -> bool {
    coords.iter().all(|&x| x >= 0.) && coords.iter().all(|&x| x <= 1.)
}

impl Triangle {
    fn get_corners(&self) -> [[f64; 2]; 2] {
        let mut c1 = [0.; 2];
        let mut c2 = [0.; 2];

        for (i, col) in self.coords.axis_iter(Axis(1)).enumerate() {
            let (min, max) = col
                .iter()
                .copied()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| {
                    (min.min(v), max.max(v))
                });
            c1[i] = min;
            c2[i] = max;
        }
        [c1, c2]
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
}

fn make_index(x: Array1<f64>, y: Array1<f64>, ikle: Vec<[usize; 3]>) -> RTree<Triangle> {
    let elements = ikle
        .iter()
        .map(|edges| Triangle {
            coords: arr2(&[
                [x[edges[0]], y[edges[0]]],
                [x[edges[1]], y[edges[1]]],
                [x[edges[2]], y[edges[2]]],
            ]),
            indices: edges.clone(),
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

    fn get_intersecting_elements(&self, point: [f64; 2]) -> Vec<&Triangle> {
        self.rtree.locate_all_at_point(&point).collect()
    }

    fn get_point_interpolators(&self, points: Vec<[f64; 2]>) -> Vec<Option<CoordResult>> {
        // points.iter()
        // .map(|point| self.get_intersecting_elements(point))
        // .map(|e| )

        let mut interpolators = Vec::<Option<CoordResult>>::new();

        for point in points {
            let potential_elements = self.get_intersecting_elements(point);
            if potential_elements.is_empty() {
                interpolators.push(None);
                continue;
            }
            let interpolator = potential_elements
                .iter()
                .map(|tri| CoordResult {
                    indices: tri.indices,
                    coords: tri.barycentric_coordinates(&point),
                })
                .find(|coord| is_inside(&coord.coords));
            interpolators.push(interpolator)
        }
        interpolators
    }
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

    fn get_intersecting_elements<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<f64>,
    ) -> Vec<Bound<'py, PyArray2<f64>>> {
        let point = point.as_array().to_owned();
        self.index
            .get_intersecting_elements([point[0], point[1]])
            .into_iter()
            .map(|tri| tri.coords.to_pyarray_bound(py))
            .collect()
    }

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
