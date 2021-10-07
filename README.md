![test status](https://github.com/tylerjereddy/rusty_hausdorff/actions/workflows/linux.yml/badge.svg)

This is a Rust implementation of the directed [Hausdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance).
It is currently intended as an experiment to see if it can be
built to outperform the SciPy implementation of [`directed_hausdorff()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html)
by leveraging i.e., safe concurrency in Rust.

Initial performance comparison with the serial SciPy
implementation [shows substantial performance improvements](https://github.com/scipy/scipy/issues/14719)
with the parallel Rust code in this project.
