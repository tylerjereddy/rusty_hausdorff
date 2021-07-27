This is a Rust implementation of the directed [Hausdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance).
It is currently intended as an experiment to see if it can be
built to outperform the SciPy implementation of [`directed_hausdorff()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html)
by leveraging i.e., safe concurrency in Rust.
