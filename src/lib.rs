use ndarray::Array2;

pub fn directed_hausdorff(ar1: Array2<f64>, ar2: Array2<f64>) -> (f64, usize, usize)
{
    
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
        let a1 = arr2(&[[1., 2., 3.],
                        [4., 5., 6.]]);
        let a2 = arr2(&[[1., 2., 3.],
                        [4., 5., 6.]]);
        // NOTE: this is the same result as
        // directed_hausdorff(arr, arr, seed=1)
        // but not seed=0, which is ok, for now
        assert_eq!(directed_hausdorff(a1, a2), (0.0, 1, 1));
    }
}

#[cfg(test)]
mod scipy_tests {
    // test for behavior similar to the SciPy
    // directed_hausdorff implementation
    // make a `NOTE` in cases where there is a devation
    // due to the random shuffling/seed in SciPy
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_indices_scipy() {
        // test for a result identical to SciPy test:
        // test_hausdorff.py::TestHausdorff::test_indices
        let path_simple_1 = arr2(&[[-1.,-12.],[0.,0.], [1.,1.], [3.,7.], [1.,2.]]);
        let path_simple_2 = arr2(&[[0.,0.], [1.,1.], [4.,100.], [10.,9.]]);
        let expected_result = (93.00537618869137, 2, 3);
        let actual_result = directed_hausdorff(path_simple_2, path_simple_1);
        assert_eq!(actual_result, expected_result);
    }
}
