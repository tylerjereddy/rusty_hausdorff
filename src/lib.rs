use ndarray::Array2;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

pub fn directed_hausdorff(
    ar1: Arc<Array2<f64>>,
    ar2: Arc<Array2<f64>>,
    workers: u64,
) -> (f64, usize, usize) {
    // TODO: decide if we're going to use random
    // shuffling in the Rust version to match
    // the SciPy implementation?

    if workers == 0 {
        // single thread/serial approach
        directed_hausdorff_core(&ar1, &ar2)
    } else {
        let (tx, rx) = mpsc::channel();
        for _ in 0..workers {
            // TODO: parallel implementation where
            // each spawned thread works on a subset
            // of ar1 instead of all threads doing
            // the same work...
            let sub_tx = tx.clone();
            let arr1 = ar1.clone();
            let arr2 = ar2.clone();
            thread::spawn(move || {
                let thread_result = directed_hausdorff_core(&arr1, &arr2);
                sub_tx.send(thread_result).unwrap();
            });
        }
        rx.recv().unwrap()
    }
}

fn directed_hausdorff_core(ar1: &Array2<f64>, ar2: &Array2<f64>) -> (f64, usize, usize) {
    let mut cmax = 0.0;
    let mut d = 0.0;
    let num_dims = ar1.shape()[1];
    let mut i_store = 0;
    let mut j_store = 0;
    let mut i_ret = 0;
    let mut j_ret = 0;

    for (i, row_i) in ar1.outer_iter().enumerate() {
        let mut cmin = f64::INFINITY;
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

fn _tuple_sorter() {
    // this function should accept a data structure
    // that contains the (cmax, i, j) tuples from parallel
    // threads and returns the true directed hausdorff
    // tuple, which would be the one with largest cmax
    // so, some kind of sorting work
    // NOTE: have not decided how the tuples will
    // be stored as a group just yet
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
        assert_eq!(
            directed_hausdorff(Arc::new(a1), Arc::new(a2), 0),
            (0.0, 1, 1)
        );
    }

    #[test]
    fn compare_scipy_2d() {
        // this isn't part of the SciPy test suite, but
        // spot check this matching result for a random
        // 2D case
        let a1 = Arc::new(arr2(&[[0., 0.7], [1., 6.5], [0., 17.], [-1., 9.]]));
        let a2 = Arc::new(arr2(&[[77., 7.2], [15., 5.5], [-9., 16.]]));
        let expected = (15.749285698088025, 0, 1);
        let actual = directed_hausdorff(a1.clone(), a2.clone(), 0);
        assert_eq!(actual, expected);
        let expected_reverse = (76.00322361584408, 0, 1);
        let actual_reverse = directed_hausdorff(a2.clone(), a1.clone(), 0);
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
        let a1 = Arc::new(a1);
        let a2 = Arc::new(a2);
        let actual = directed_hausdorff(a1.clone(), a2.clone(), 0);
        let actual_reverse = directed_hausdorff(a2.clone(), a1.clone(), 0);
        assert_eq!(actual, expected);
        assert_eq!(actual_reverse, expected_reverse);
    }

    #[test]
    fn compare_scipy_4d() {
        // just a "random" 4D comparison against
        // SciPy directed_hausdorff()
        let a1 = arr2(&[
            [0.28831009, 2.58107231, 2.55713349, 0.99507485],
            [1.84067156, 1.12988751, 0.48848989, 2.33141212],
            [2.26440714, 0.73540113, 1.79867459, 0.5349007],
            [2.47988248, 0.40452796, 2.20779778, 2.75998248],
            [2.97570666, 1.33849502, 0.51443716, 0.1457924],
            [0.62640827, 0.32842044, 0.08219697, 0.45037708],
            [1.49108549, 0.43263036, 1.61092912, 0.62350793],
            [2.38296706, 0.57023688, 1.47690286, 1.17498774],
        ]);
        let a2 = arr2(&[
            [2.44051647, 3.72771419, 1.70998132, 3.27594938],
            [0.56762581, 1.09420038, 2.93499868, 2.19989798],
            [1.79280814, 1.92535183, 0.207613, 2.32548235],
            [3.30735955, 0.76909727, 3.25351285, 2.26041119],
            [2.67056764, 1.63746358, 3.70897313, 2.07159087],
            [2.64122702, 0.90337615, 3.38465142, 1.65059944],
            [1.34034943, 1.99235183, 2.26215791, 3.11249246],
            [1.4348432, 3.25730841, 3.66045435, 2.61105284],
            [2.8803127, 3.50204148, 2.91110745, 0.72640875],
            [2.3430369, 3.82914243, 2.97951285, 2.09052481],
            [3.3004003, 2.85735617, 1.23357079, 1.30450906],
        ]);
        let expected = (2.728081280912664, 5, 2);
        let expected_reverse = (3.081024070737598, 0, 1);
        let a1 = Arc::new(a1);
        let a2 = Arc::new(a2);
        let actual = directed_hausdorff(a1.clone(), a2.clone(), 0);
        let actual_reverse = directed_hausdorff(a2.clone(), a1.clone(), 0);
        assert_eq!(actual, expected);
        assert_eq!(actual_reverse, expected_reverse);
    }

    #[test]
    fn compare_scipy_5d() {
        // just a "random" 5D comparison against
        // SciPy directed_hausdorff()
        let a1 = arr2(&[
            [2.9151137, 15.03350375, 1.4054412, 13.2398229, 10.49955955],
            [15.3465167, 12.18641521, 9.57434635, 12.15742319, 1.52864431],
            [10.37315124, 6.60944447, 8.77464511, 14.23145089, 9.09959121],
            [
                14.52474052,
                15.48086054,
                11.26433807,
                16.03428425,
                4.7060331,
            ],
            [9.23022243, 3.94966952, 3.47835833, 13.2534381, 1.89727603],
            [0.52856656, 6.96496597, 3.91919165, 6.55635451, 8.42974687],
            [4.49494713, 14.00054623, 5.31482836, 7.71453595, 11.55373822],
            [4.88268573, 12.87529517, 3.25030969, 1.81119906, 4.51157337],
            [0.46565545, 11.79210413, 7.91756568, 4.5690695, 2.50492342],
            [
                13.22037192,
                11.56444441,
                13.06148135,
                4.79015061,
                9.22399885,
            ],
            [3.74150036, 4.78944612, 2.17618963, 16.24449808, 12.28701247],
            [13.17573659, 10.18563589, 6.83190693, 4.08871476, 9.62830105],
            [7.87640769, 1.23909591, 1.6360347, 11.35071203, 6.56404739],
            [8.65027547, 0.56682667, 8.20501381, 15.81005849, 7.03054853],
            [8.93148643, 4.08471583, 13.32701046, 7.76561772, 7.03277901],
            [3.30002915, 11.6170999, 7.95974807, 15.98257198, 16.70219885],
            [8.44289226, 5.82440432, 13.58966062, 16.06989395, 1.24406946],
            [0.16019919, 16.1379727, 2.3481645, 9.05804532, 4.90200271],
            [4.5613058, 16.35441708, 0.23315048, 5.61515763, 5.54930709],
            [15.07675337, 0.5694987, 3.23436396, 2.75091626, 12.53727234],
            [12.07910137, 9.08885975, 16.92424106, 0.0252033, 8.28304905],
            [7.97339419, 12.46487871, 2.95079891, 6.92334564, 4.17006241],
            [15.29105719, 5.22582819, 0.48859639, 2.58302021, 12.8937161],
            [10.41449398, 4.73238648, 5.03681252, 16.90130974, 4.31555678],
            [15.14729305, 14.86132495, 0.04709327, 7.90313308, 9.98387264],
            [8.99683656, 12.72901207, 2.49699729, 1.2032349, 7.58543564],
            [1.50657266, 6.34333631, 5.40486632, 8.03437984, 0.17551712],
        ]);
        let a2 = arr2(&[
            [0.14919222, 0.39827536, 0.62668944, 2.59954114, 2.19178342],
            [0.74678694, 2.79557876, 3.00658436, 1.28402571, 3.20835286],
            [1.91003573, 0.98594454, 1.66764679, 0.31088887, 2.19388982],
            [3.60096208, 2.02501205, 2.5332713, 0.03273345, 0.0214961],
            [3.17985577, 1.55459124, 1.44510951, 0.18145603, 2.69697106],
            [2.16478871, 3.37317003, 1.03044564, 3.85575298, 2.0516388],
            [1.4451347, 1.04206897, 2.38772263, 2.1688008, 1.96335907],
            [0.47596543, 0.12269475, 3.12026109, 3.44302907, 1.08801091],
            [0.52144018, 0.3525681, 3.82693826, 2.08640732, 3.29976525],
            [2.75763369, 0.2657139, 2.26311721, 2.80529272, 0.2507785],
            [1.47326588, 3.83013821, 0.7846098, 1.63853832, 0.17657007],
        ]);
        let expected = (23.652897613073076, 3, 5);
        let expected_reverse = (10.461250978884332, 4, 26);
        let a1 = Arc::new(a1);
        let a2 = Arc::new(a2);
        let actual = directed_hausdorff(a1.clone(), a2.clone(), 0);
        let actual_reverse = directed_hausdorff(a2.clone(), a1.clone(), 0);
        assert_eq!(actual, expected);
        assert_eq!(actual_reverse, expected_reverse);
    }
    #[test]
    fn compare_scipy_6d() {
        // just a "random" 6D comparison against
        // SciPy directed_hausdorff()
        let a1 = arr2(&[
            [
                1.46996317, 1.30327733, 0.71454121, 0.13462983, 1.46610658, 1.92399224,
            ],
            [
                0.16419724, 1.64391755, 0.19506457, 0.92118181, 1.11139359, 0.28538699,
            ],
            [
                0.98293502, 0.1193791, 0.98106616, 1.44663558, 1.6967336, 0.10112273,
            ],
            [
                0.35212957, 1.57295086, 1.76855715, 1.88844348, 1.71008165, 0.32143576,
            ],
            [
                1.66301731, 0.82833205, 1.44143695, 0.80624939, 1.11048028, 0.51241501,
            ],
            [
                1.10180818, 0.39221057, 1.86754752, 0.45514167, 0.51881934, 1.17488038,
            ],
            [
                1.60015109, 1.71067643, 0.51571922, 1.21001692, 1.53763651, 1.74959373,
            ],
            [
                1.48914803, 0.84244971, 1.656104, 1.6473143, 0.33177637, 0.81133061,
            ],
            [
                1.67726085, 0.26294712, 1.82206166, 0.69504977, 1.70551739, 0.35433391,
            ],
            [
                0.71787631, 1.13632226, 1.25699597, 0.15711922, 1.12881012, 0.97035123,
            ],
            [
                0.44449654, 1.82777046, 1.71813115, 1.97506096, 1.40872443, 0.47097485,
            ],
            [
                1.58222125, 1.56166706, 0.56567779, 0.64334872, 1.94206051, 1.57112395,
            ],
            [
                0.40662456, 1.47740817, 0.57906734, 1.880572, 1.12974295, 1.77180157,
            ],
            [
                1.9562735, 0.83753321, 0.44700344, 1.23652265, 0.21643231, 1.48849098,
            ],
            [
                1.29658185, 1.43616231, 0.82220356, 0.43300425, 1.41844641, 1.79011512,
            ],
        ]);
        let a2 = arr2(&[
            [
                1.60642112, 1.7415969, 0.27883749, 0.33599065, 0.2634141, 0.3304697,
            ],
            [
                0.32147681, 1.81388517, 0.33148916, 1.68995247, 1.18757631, 1.75813478,
            ],
            [
                0.79653733, 1.71872265, 0.10426738, 1.68241098, 1.22079625, 0.49692932,
            ],
            [
                1.57149378, 1.64665453, 0.25434893, 0.08611663, 1.46253612, 0.56015041,
            ],
            [
                1.34465798, 0.00199996, 1.1554701, 1.05124526, 0.45086164, 0.89902654,
            ],
            [
                0.37855275, 1.8347836, 0.40778384, 0.14821605, 1.55554427, 0.59422687,
            ],
            [
                0.77346841, 1.40438723, 0.08245654, 1.89689766, 0.08211211, 1.27643619,
            ],
            [
                0.57414328, 0.34695469, 0.99295438, 0.6420694, 0.77350822, 1.27089656,
            ],
            [
                0.20165678, 1.35285454, 0.78210969, 0.37773515, 0.69593044, 0.28913069,
            ],
            [
                1.51949293, 0.35570411, 0.58828738, 0.36529037, 0.42594443, 1.97722972,
            ],
            [
                0.09074949, 0.43460658, 1.22839255, 1.60591857, 0.74183357, 1.89551667,
            ],
            [
                0.62950348, 0.81041341, 1.44271106, 1.47256241, 1.09743396, 1.95491269,
            ],
        ]);
        let expected = (1.8169357979151486, 3, 2);
        let expected_reverse = (1.7669708788364127, 0, 13);
        let a1 = Arc::new(a1);
        let a2 = Arc::new(a2);
        let actual = directed_hausdorff(a1.clone(), a2.clone(), 0);
        let actual_reverse = directed_hausdorff(a2.clone(), a1.clone(), 0);
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
    use ndarray_npy::NpzReader;
    use std::fs::File;

    fn setup_tests() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        // make the exact arrays used in the SciPy test
        // suite available for testing here
        let mut npz =
            NpzReader::new(File::open("src/paths.npz").expect("unable to open data file"))
                .expect("unable to read from file");
        let path_1: Array2<f64> = npz
            .by_name("path_1.npy")
            .expect("unable to retrieve path_1 field");
        let path_2: Array2<f64> = npz
            .by_name("path_2.npy")
            .expect("unable to retrieve path_2 field");
        let path_1_4d: Array2<f64> = npz
            .by_name("path_1_4d.npy")
            .expect("unable to retrieve path_1_4d field");
        let path_2_4d: Array2<f64> = npz
            .by_name("path_2_4d.npy")
            .expect("unable to retrieve path_2_4d field");
        (path_1, path_2, path_1_4d, path_2_4d)
    }

    #[test]
    fn test_indices_scipy() {
        // test for a result identical to SciPy test:
        // test_hausdorff.py::TestHausdorff::test_indices
        let path_simple_1 = arr2(&[[-1., -12.], [0., 0.], [1., 1.], [3., 7.], [1., 2.]]);
        let path_simple_2 = arr2(&[[0., 0.], [1., 1.], [4., 100.], [10., 9.]]);
        let expected_result = (93.00537618869137, 2, 3);
        let actual_result = directed_hausdorff(Arc::new(path_simple_2), Arc::new(path_simple_1), 0);
        assert_eq!(actual_result, expected_result);
    }

    #[test]
    fn test_symmetry_scipy() {
        // test for a result identical to SciPy test:
        // test_hausdorff.py::TestHausdorff::test_symmetry
        let (path_1, path_2, _, _) = setup_tests();
        let expected_forward = 1.000681524361451;
        let expected_reverse = 2.3000000000000003;
        let path_1 = Arc::new(path_1);
        let path_2 = Arc::new(path_2);
        let actual_forward = directed_hausdorff(path_1.clone(), path_2.clone(), 0).0;
        let actual_reverse = directed_hausdorff(path_2.clone(), path_1.clone(), 0).0;
        assert_ne!(actual_forward, actual_reverse);
        assert_eq!(actual_forward, expected_forward);
        assert_eq!(actual_reverse, expected_reverse);
    }
    #[test]
    fn test_brute_force_comparison_forward_scipy() {
        // test for a result identical to SciPy test:
        // test_hausdorff.py::TestHausdorff::test_brute_force_comparison_forward
        let (path_1, path_2, _, _) = setup_tests();
        let expected_forward = 1.000681524361451;
        let actual_forward = directed_hausdorff(Arc::new(path_1), Arc::new(path_2), 0).0;
        assert_eq!(actual_forward, expected_forward);
    }

    #[test]
    fn test_brute_force_comparison_reverse_scipy() {
        // test for a result identical to SciPy test:
        // test_hausdorff.py::TestHausdorff::test_brute_force_comparison_reverse
        let (path_1, path_2, _, _) = setup_tests();
        let expected_reverse = 2.3000000000000003;
        let actual_reverse = directed_hausdorff(Arc::new(path_2), Arc::new(path_1), 0).0;
        assert_eq!(actual_reverse, expected_reverse);
    }

    #[test]
    fn test_degenerate_case_scipy() {
        // test for a result identical to SciPy test:
        // test_hausdorff.py::TestHausdorff::test_degenerate_case
        let (path_1, _, _, _) = setup_tests();
        let expected = 0.0;
        let path_1 = Arc::new(path_1);
        let actual = directed_hausdorff(path_1.clone(), path_1.clone(), 0).0;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_2d_data_forward_scipy() {
        // test for a result identical to SciPy test:
        // test_hausdorff.py::TestHausdorff::test_2d_data_forward
        let (path_1, path_2, _, _) = setup_tests();
        let path_1 = path_1.slice(s![.., ..2]).to_owned();
        let path_2 = path_2.slice(s![.., ..2]).to_owned();
        let expected = 1.000681524361451;
        let actual = directed_hausdorff(Arc::new(path_1), Arc::new(path_2), 0).0;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_4d_data_reverse_scipy() {
        // test for a result identical to SciPy test:
        // test_hausdorff.py::TestHausdorff::test_4d_data_reverse
        let (_, _, path_1_4d, path_2_4d) = setup_tests();
        let expected = 22.119900542271886;
        let path_1_4d = Arc::new(path_1_4d);
        let path_2_4d = Arc::new(path_2_4d);
        for workers in 0..4 {
            let actual = directed_hausdorff(path_2_4d.clone(), path_1_4d.clone(), workers).0;
            assert_eq!(actual, expected);
        }
    }
}
