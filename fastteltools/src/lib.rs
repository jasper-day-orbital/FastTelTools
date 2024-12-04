use ndarray::{arr1, arr2, stack, Array1, Array2, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{prelude::*, pyclass};
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};

#[derive(Clone, Debug)]
struct Triangle {
    coords: Array2<f32>,
    indices: Array1<i32>,
    corners: [[f32; 2]; 2],
}

#[inline]
fn is_inside(coords: &Array1<f32>) -> bool {
    coords.iter().all(|&x| x >= 0. && x <= 1.)
}

fn get_corners(coords: &Array2<f32>) -> [[f32; 2]; 2] {
    let mut c1 = [0.; 2];
    let mut c2 = [0.; 2];

    for (i, col) in coords.axis_iter(Axis(1)).enumerate() {
        let (min, max) = col
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), v| {
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
    fn new(coords: Array2<f32>, indices: Array1<i32>) -> Triangle {
        Triangle {
            corners: get_corners(&coords),
            coords: coords,
            indices: indices,
        }
    }

    fn get_corners(&self) -> [[f32; 2]; 2] {
        self.corners
    }

    fn barycentric_coordinates(&self, point: &[f32; 2]) -> Array1<f32> {
        let t = &self.coords;
        let x = point[0];
        let y = point[1];
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
}

impl RTreeObject for Triangle {
    type Envelope = AABB<[f32; 2]>;

    fn envelope(&self) -> Self::Envelope {
        let corners = self.get_corners();
        AABB::from_corners(corners[0], corners[1])
    }
}

impl PointDistance for Triangle {
    fn distance_2(&self, point: &[f32; 2]) -> f32 {
        self.envelope().distance_2(point)
    }

    fn contains_point(&self, point: &<Self::Envelope as rstar::Envelope>::Point) -> bool {
        is_inside(&self.barycentric_coordinates(point))
    }
}

fn make_index(x: Array1<f32>, y: Array1<f32>, ikle: Array2<i32>) -> RTree<Triangle> {
    let elements = ikle
        .rows()
        .into_iter()
        .map(|edges| {
            Triangle::new(
                arr2(&[
                    [x[edges[0] as usize], y[edges[0] as usize]],
                    [x[edges[1] as usize], y[edges[1] as usize]],
                    [x[edges[2] as usize], y[edges[2] as usize]],
                ]),
                edges.clone().to_owned(),
            )
        })
        .collect();
    RTree::bulk_load(elements)
}

#[derive(Debug)]
struct Mesh2D {
    points: Array2<f32>,
    rtree: RTree<Triangle>,
}

impl Mesh2D {
    fn new(x: Array1<f32>, y: Array1<f32>, ikle: Array2<i32>) -> Mesh2D {
        Mesh2D {
            points: stack![Axis(0), x, y],
            rtree: make_index(x, y, ikle),
        }
    }

    fn locate_at_point(&self, point: &[f32; 2]) -> Option<&Triangle> {
        self.rtree.locate_at_point(point)
    }

    fn collect_result(
        &self,
        result: Vec<Option<(Array1<f32>, &Array1<i32>)>>,
    ) -> (Array2<f32>, Array2<i32>) {
        // Turn a vector of partial results into a single array of results
        let npoints = result.len();
        let mut out_coords = Array2::<f32>::zeros((npoints, 3));
        out_coords.fill(f32::NAN);
        let mut out_inds = Array2::<i32>::zeros((npoints, 3));
        out_inds.fill(-1);
        for (i, res) in result.into_iter().enumerate() {
            if let Some((coords, inds)) = res {
                out_coords.row_mut(i).assign(&coords);
                out_inds.row_mut(i).assign(inds);
            }
        }
        (out_coords, out_inds)
    }

    fn get_point_interpolators(&self, points: Array2<f32>) -> (Array2<f32>, Array2<i32>) {
        let result = points
            .rows()
            .into_iter()
            .map(|row| {
                let point = &[row[0], row[1]];
                match self.locate_at_point(point) {
                    Some(tri) => Some((tri.barycentric_coordinates(point), &tri.indices)),
                    None => None,
                }
            })
            .collect();
        self.collect_result(result)
    }

    fn get_point_interpolators_parallel(&self, points: Array2<f32>) -> (Array2<f32>, Array2<i32>) {
        let npoints = points.len_of(Axis(0));

        let mut result: Vec<Option<(Array1<f32>, &Array1<i32>)>> = vec![None; npoints];

        // We expect this to be expensive (looking up values in the rtree)
        result.par_iter_mut().enumerate().for_each(|(i, res)| {
            let point = &[points[[i, 0]], points[[i, 1]]];
            if let Some(result) = self.locate_at_point(point) {
                *res = Some((result.barycentric_coordinates(point), &result.indices))
            }
        });

        self.collect_result(result)
    }
}

#[pyclass(subclass)]
struct _PyMesh2D {
    index: Mesh2D,
}

#[pymethods]
impl _PyMesh2D {
    #[new]
    fn new(
        x: PyReadonlyArray1<f32>,
        y: PyReadonlyArray1<f32>,
        ikle: PyReadonlyArray2<i32>,
    ) -> _PyMesh2D {
        let x = x.as_array().to_owned();
        let y = y.as_array().to_owned();
        let ikle = ikle.as_array().to_owned();
        _PyMesh2D {
            index: Mesh2D::new(x, y, ikle),
        }
    }

    #[getter]
    fn points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        self.index.points.to_pyarray_bound(py)
    }

    fn get_point_interpolators<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<f32>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i32>>) {
        let points = points.as_array().to_owned();
        let interpolators = self.index.get_point_interpolators(points);
        (
            interpolators.0.to_pyarray_bound(py),
            interpolators.1.to_pyarray_bound(py),
        )
    }

    /// Optimized parallel point interpolators
    fn get_point_interpolators_parallel<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<f32>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i32>>) {
        let points = points.as_array().to_owned();
        let interpolators = self.index.get_point_interpolators_parallel(points);
        (
            interpolators.0.to_pyarray_bound(py),
            interpolators.1.to_pyarray_bound(py),
        )
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _fastteltools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<_PyMesh2D>()?;
    Ok(())
}
