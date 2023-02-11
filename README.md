Comparing various **per-thread duty** numbers for *CUDA based vector multiply*.

Two floating-point vectors `x` and `y`, with number of **elements** from
`1E+6` to `1E+9` were multiplied using CUDA. Each element count was attempted
with various **CUDA launch configs** and **per-thread duties**, running each
config 5 times to get a good time measure. Multiplication here represents any
memory-aligned independent operation, or a `map()` operation. Results indicate
no significant difference between [launch adjust] approach, and this one.

All outputs for [float] and [double] are saved in *gists* and a small part of
the output is listed here. This experiment was done with guidance from
[Prof. Dip Sankar Banerjee] and [Prof. Kishore Kothapalli].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# ...
#
# # Elements 1e+07
# [00002.143 ms] [1.644725] multiplySeq
# [00000.181 ms] [1.644725] multiplyCuda<<<auto, 32>>> [thread-duty=1]
# [00000.179 ms] [1.644725] multiplyCuda<<<auto, 32>>> [thread-duty=2]
# [00000.180 ms] [1.644725] multiplyCuda<<<auto, 32>>> [thread-duty=3]
# [00000.181 ms] [1.644725] multiplyCuda<<<auto, 32>>> [thread-duty=4]
# [00000.178 ms] [1.644725] multiplyCuda<<<auto, 32>>> [thread-duty=6]
# [00000.179 ms] [1.644725] multiplyCuda<<<auto, 32>>> [thread-duty=8]
# [00000.183 ms] [1.644725] multiplyCuda<<<auto, 32>>> [thread-duty=12]
# [00000.182 ms] [1.644725] multiplyCuda<<<auto, 32>>> [thread-duty=16]
# [00000.188 ms] [1.644725] multiplyCuda<<<auto, 32>>> [thread-duty=24]
# [00000.178 ms] [1.644725] multiplyCuda<<<auto, 32>>> [thread-duty=32]
# [00000.183 ms] [1.644725] multiplyCuda<<<auto, 32>>> [thread-duty=48]
# [00000.180 ms] [1.644725] multiplyCuda<<<auto, 32>>> [thread-duty=64]
# [00000.166 ms] [1.644725] multiplyCuda<<<auto, 64>>> [thread-duty=1]
# [00000.166 ms] [1.644725] multiplyCuda<<<auto, 64>>> [thread-duty=2]
# [00000.167 ms] [1.644725] multiplyCuda<<<auto, 64>>> [thread-duty=3]
# [00000.167 ms] [1.644725] multiplyCuda<<<auto, 64>>> [thread-duty=4]
# [00000.167 ms] [1.644725] multiplyCuda<<<auto, 64>>> [thread-duty=6]
# [00000.169 ms] [1.644725] multiplyCuda<<<auto, 64>>> [thread-duty=8]
# [00000.543 ms] [1.644725] multiplyCuda<<<auto, 64>>> [thread-duty=12]
# [00000.163 ms] [1.644725] multiplyCuda<<<auto, 64>>> [thread-duty=16]
# [00000.173 ms] [1.644725] multiplyCuda<<<auto, 64>>> [thread-duty=24]
# [00000.169 ms] [1.644725] multiplyCuda<<<auto, 64>>> [thread-duty=32]
# [00000.176 ms] [1.644725] multiplyCuda<<<auto, 64>>> [thread-duty=48]
# [00000.168 ms] [1.644725] multiplyCuda<<<auto, 64>>> [thread-duty=64]
# [00000.166 ms] [1.644725] multiplyCuda<<<auto, 128>>> [thread-duty=1]
# [00000.166 ms] [1.644725] multiplyCuda<<<auto, 128>>> [thread-duty=2]
# [00000.166 ms] [1.644725] multiplyCuda<<<auto, 128>>> [thread-duty=3]
# [00000.166 ms] [1.644725] multiplyCuda<<<auto, 128>>> [thread-duty=4]
# [00000.168 ms] [1.644725] multiplyCuda<<<auto, 128>>> [thread-duty=6]
# [00000.169 ms] [1.644725] multiplyCuda<<<auto, 128>>> [thread-duty=8]
# [00000.170 ms] [1.644725] multiplyCuda<<<auto, 128>>> [thread-duty=12]
# [00000.169 ms] [1.644725] multiplyCuda<<<auto, 128>>> [thread-duty=16]
# [00000.174 ms] [1.644725] multiplyCuda<<<auto, 128>>> [thread-duty=24]
# [00000.168 ms] [1.644725] multiplyCuda<<<auto, 128>>> [thread-duty=32]
# [00000.177 ms] [1.644725] multiplyCuda<<<auto, 128>>> [thread-duty=48]
# [00000.167 ms] [1.644725] multiplyCuda<<<auto, 128>>> [thread-duty=64]
# [00000.166 ms] [1.644725] multiplyCuda<<<auto, 256>>> [thread-duty=1]
# [00000.166 ms] [1.644725] multiplyCuda<<<auto, 256>>> [thread-duty=2]
# [00000.168 ms] [1.644725] multiplyCuda<<<auto, 256>>> [thread-duty=3]
# [00000.166 ms] [1.644725] multiplyCuda<<<auto, 256>>> [thread-duty=4]
# [00000.168 ms] [1.644725] multiplyCuda<<<auto, 256>>> [thread-duty=6]
# [00000.168 ms] [1.644725] multiplyCuda<<<auto, 256>>> [thread-duty=8]
# [00000.171 ms] [1.644725] multiplyCuda<<<auto, 256>>> [thread-duty=12]
# [00000.170 ms] [1.644725] multiplyCuda<<<auto, 256>>> [thread-duty=16]
# [00000.173 ms] [1.644725] multiplyCuda<<<auto, 256>>> [thread-duty=24]
# [00000.168 ms] [1.644725] multiplyCuda<<<auto, 256>>> [thread-duty=32]
# [00000.179 ms] [1.644725] multiplyCuda<<<auto, 256>>> [thread-duty=48]
# [00000.167 ms] [1.644725] multiplyCuda<<<auto, 256>>> [thread-duty=64]
# [00000.166 ms] [1.644725] multiplyCuda<<<auto, 512>>> [thread-duty=1]
# [00000.166 ms] [1.644725] multiplyCuda<<<auto, 512>>> [thread-duty=2]
# [00000.167 ms] [1.644725] multiplyCuda<<<auto, 512>>> [thread-duty=3]
# [00000.166 ms] [1.644725] multiplyCuda<<<auto, 512>>> [thread-duty=4]
# [00000.167 ms] [1.644725] multiplyCuda<<<auto, 512>>> [thread-duty=6]
# [00000.169 ms] [1.644725] multiplyCuda<<<auto, 512>>> [thread-duty=8]
# [00000.171 ms] [1.644725] multiplyCuda<<<auto, 512>>> [thread-duty=12]
# [00000.169 ms] [1.644725] multiplyCuda<<<auto, 512>>> [thread-duty=16]
# [00000.173 ms] [1.644725] multiplyCuda<<<auto, 512>>> [thread-duty=24]
# [00000.170 ms] [1.644725] multiplyCuda<<<auto, 512>>> [thread-duty=32]
# [00000.178 ms] [1.644725] multiplyCuda<<<auto, 512>>> [thread-duty=48]
# [00000.167 ms] [1.644725] multiplyCuda<<<auto, 512>>> [thread-duty=64]
# [00000.168 ms] [1.644725] multiplyCuda<<<auto, 1024>>> [thread-duty=1]
# [00000.166 ms] [1.644725] multiplyCuda<<<auto, 1024>>> [thread-duty=2]
# [00000.167 ms] [1.644725] multiplyCuda<<<auto, 1024>>> [thread-duty=3]
# [00000.167 ms] [1.644725] multiplyCuda<<<auto, 1024>>> [thread-duty=4]
# [00000.168 ms] [1.644725] multiplyCuda<<<auto, 1024>>> [thread-duty=6]
# [00000.169 ms] [1.644725] multiplyCuda<<<auto, 1024>>> [thread-duty=8]
# [00000.171 ms] [1.644725] multiplyCuda<<<auto, 1024>>> [thread-duty=12]
# [00000.169 ms] [1.644725] multiplyCuda<<<auto, 1024>>> [thread-duty=16]
# [00000.173 ms] [1.644725] multiplyCuda<<<auto, 1024>>> [thread-duty=24]
# [00000.167 ms] [1.644725] multiplyCuda<<<auto, 1024>>> [thread-duty=32]
# [00000.179 ms] [1.644725] multiplyCuda<<<auto, 1024>>> [thread-duty=48]
# [00000.167 ms] [1.644725] multiplyCuda<<<auto, 1024>>> [thread-duty=64]
#
# ...
```

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://www.slideshare.net/SubhajitSahu/cuda-by-example-notes)

<br>
<br>

[![](https://i.imgur.com/KExwVG1.jpg)](https://www.youtube.com/watch?v=A7TKQKAFIi4)
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/413813514.svg)](https://zenodo.org/badge/latestdoi/413813514)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[launch adjust]: https://github.com/puzzlef/vector-multiply-cuda
[float]: https://gist.github.com/wolfram77/6b68c212bc06f67d8afb18270e8865a8
[double]: https://gist.github.com/wolfram77/fb5280044ca2d844158c36ce2afebdf0
