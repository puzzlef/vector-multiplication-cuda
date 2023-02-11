Comparing approaches for *CUDA-based* **vector multiplication**.

In each of the experiments given below, we multiply two floating-point vectors
`x` and `y`, with number of **elements** from `10^6` to `10^9` using OpenMP.
Each element count is attempted with various approaches, running each approach 5
times to get a good time measure. Multiplication here represents any
memory-aligned independent operation, or a `map()` operation.

<br>


### Adjusting Launch config

In this experiment ([adjust-launch]), we multiply two floating-point vectors `x`
and `y` using CUDA. Each element count is attempted with various **CUDA launch**
**configs**. Results indicate that a **grid_limit** of `16384/32768`, and a
**block_size** of `128/256` to be suitable for both **float** and **double**.
Using a **grid_limit** of `MAX` and a **block_size** of `256` could be a decent
choice.

[adjust-launch]: https://github.com/puzzlef/vector-multiplication-cuda/tree/adjust-launch

<br>


### Adjusting Thread duty

In this experiment ([adjust-duty]), we compare various *per-thread duty numbers*
for CUDA-based vector multiplication. Each element count is attempted with
various CUDA launch configs and per-thread duties. Results indicate no
significant difference between [adjust-launch] approach, and this one.

[adjust-duty]: https://github.com/puzzlef/vector-multiplication-cuda/tree/adjust-duty

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://www.slideshare.net/SubhajitSahu/cuda-by-example-notes)
- [Git pulling a branch from another repository?](https://stackoverflow.com/a/46289324/1413259)

<br>
<br>


[![](https://i.imgur.com/azEBS7Y.png)](https://www.youtube.com/watch?v=vTdodyhhjww)
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/375073607.svg)](https://zenodo.org/badge/latestdoi/375073607)


[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
