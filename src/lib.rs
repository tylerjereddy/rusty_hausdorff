use ndarray::Array2;

pub fn directed_hausdorff(ar1: Array2<f64>, ar2: Array2<f64>) -> f64
{
    
    let inf = f64::INFINITY;
    let mut cmax = 0.0;
    let mut d = 0.0;
    let num_dims = ar1.shape()[1];
    
	for row_i in ar1.outer_iter() {
        let mut cmin = inf;
        for row_j in ar2.outer_iter() {
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
            }
        }
    if cmin >= cmax && d >= cmax {
        cmax = cmin;
    }
    }
d.sqrt()
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
        assert_eq!(directed_hausdorff(a1, a2), 0.0);
    }
}
