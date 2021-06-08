use ndarray::Array2;

pub fn directed_hausdorff(ar1: &Array2<f64>, ar2: &Array2<f64>) -> (f64, usize, usize) {
    let inf = f64::INFINITY;
    let mut cmax = 0.0;
    let mut d = 0.0;
    let num_dims = ar1.shape()[1];
    let mut i_store = 0;
    let mut j_store = 0;
    let mut i_ret = 0;
    let mut j_ret = 0;

    // TODO: decide if we're going to use random
    // shuffling in the Rust version to match
    // the SciPy implementation?

    for (i, row_i) in ar1.outer_iter().enumerate() {
        let mut cmin = inf;
        for (j, row_j) in ar2.outer_iter().enumerate() {
            d = 0.0;
            for dim in 0..num_dims {
                // square of distance -- avoid sqrt
                // until very end for performance
                d += (row_i[dim] - row_j[dim]).powi(2);
            }
            if d < cmax {
                break;
            }
            if d < cmin {
                cmin = d;
                i_store = i;
                j_store = j;
            }
        }
        if cmin >= cmax && d >= cmax {
            cmax = cmin;
            i_ret = i_store;
            j_ret = j_store;
        }
    }
    (cmax.sqrt(), i_ret, j_ret)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;
    #[test]
    fn identical_arrays() {
        // the directed Hausdorff distance between
        // identical arrays should always be zero
        let a1 = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        let a2 = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        // NOTE: this is the same result as SciPy
        // directed_hausdorff(arr, arr, seed=1)
        // but not seed=0, which is ok, for now
        assert_eq!(directed_hausdorff(&a1, &a2), (0.0, 1, 1));
    }

    #[test]
    fn compare_scipy_2d() {
        // this isn't part of the SciPy test suite, but
        // spot check this matching result for a random
        // 2D case
        let a1 = arr2(&[[0., 0.7], [1., 6.5], [0., 17.], [-1., 9.]]);
        let a2 = arr2(&[[77., 7.2], [15., 5.5], [-9., 16.]]);
        let expected = (15.749285698088025, 0, 1);
        let actual = directed_hausdorff(&a1, &a2);
        assert_eq!(actual, expected);
        let expected_reverse = (76.00322361584408, 0, 1);
        let actual_reverse = directed_hausdorff(&a2, &a1);
        assert_eq!(actual_reverse, expected_reverse);
    }
    #[test]
    fn compare_scipy_3d() {
        // this isn't part of the SciPy test suite, but
        // spot check this matching result for a random
        // 3D case
        let a1 = arr2(&[
            [1.74817113, 0.27479074, 4.65646198],
            [1.16174122, 2.13268615, 3.23624532],
            [0.91969822, 4.20465196, 1.15848725],
            [4.48290033, 2.91773345, 2.61951886],
            [4.57078972, 1.74358378, 4.15133407],
            [0.17069513, 3.31458363, 0.52684683],
            [0.84210648, 0.53257252, 4.18936828],
            [1.6894125, 4.62749626, 1.12010661],
            [0.72604676, 3.57036911, 4.59478599],
            [3.9814702, 1.99103389, 1.58619867],
        ]);
        let a2 = arr2(&[
            [-16.81962165, -12.78575936, -16.13888657],
            [-10.19793567, -14.37783032, -16.37554908],
            [-14.55197831, -13.00309863, -11.39132725],
            [-13.99939045, -12.25764784, -11.4568613],
            [-12.93828217, -13.7891231, -12.30778668],
            [-16.64497264, -10.81115498, -12.19459171],
            [-15.76043418, -15.68612392, -12.44984599],
            [-13.22091713, -10.50824843, -10.86545019],
            [-13.20634447, -10.66039683, -11.04265443],
            [-12.22670677, -15.27220865, -14.27289413],
        ]);
        let expected = (26.308858482452525, 4, 7);
        let expected_reverse = (28.733927306239696, 0, 5);
        let actual = directed_hausdorff(&a1, &a2);
        let actual_reverse = directed_hausdorff(&a2, &a1);
        assert_eq!(actual, expected);
        assert_eq!(actual_reverse, expected_reverse);
    }
}

#[cfg(test)]
mod scipy_tests {
    // test for behavior similar to the SciPy
    // directed_hausdorff implementation
    // make a `NOTE` in cases where there is a deviation
    // due to the random shuffling/seed in SciPy
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_indices_scipy() {
        // test for a result identical to SciPy test:
        // test_hausdorff.py::TestHausdorff::test_indices
        let path_simple_1 = arr2(&[[-1., -12.], [0., 0.], [1., 1.], [3., 7.], [1., 2.]]);
        let path_simple_2 = arr2(&[[0., 0.], [1., 1.], [4., 100.], [10., 9.]]);
        let expected_result = (93.00537618869137, 2, 3);
        let actual_result = directed_hausdorff(&path_simple_2, &path_simple_1);
        assert_eq!(actual_result, expected_result);
    }
}
