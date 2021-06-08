Comparing various launch configs for CUDA based vector multiply.

Two floating-point vector `x` and `y`, with no. of **elements** `1E+6` to
`1E+9` were multiplied using CUDA. Each no. of elements was attempted with
various **CUDA launch configs**, running each config 5 times to get a good
time measure. Multiplication here represents any memory-aligned independent
operation. Using a **large** `grid_limit` and a `block_size` of **256** could
be a decent choice.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# ...
#
# # Elements 1e+09
# [00279.861 ms] [1.644725] multiplySeq
# [00021.842 ms] [1.644725] multiplyCuda<<<1024, 32>>>
# [00017.829 ms] [1.644725] multiplyCuda<<<1024, 64>>>
# [00016.707 ms] [1.644725] multiplyCuda<<<1024, 128>>>
# [00016.216 ms] [1.644725] multiplyCuda<<<1024, 256>>>
# [00016.468 ms] [1.644725] multiplyCuda<<<1024, 512>>>
# [00015.225 ms] [1.644725] multiplyCuda<<<1024, 1024>>>
# [00017.614 ms] [1.644725] multiplyCuda<<<2048, 32>>>
# [00016.934 ms] [1.644725] multiplyCuda<<<2048, 64>>>
# [00016.197 ms] [1.644725] multiplyCuda<<<2048, 128>>>
# [00016.401 ms] [1.644725] multiplyCuda<<<2048, 256>>>
# [00015.286 ms] [1.644725] multiplyCuda<<<2048, 512>>>
# [00014.966 ms] [1.644725] multiplyCuda<<<2048, 1024>>>
# [00017.428 ms] [1.644725] multiplyCuda<<<4096, 32>>>
# [00016.449 ms] [1.644725] multiplyCuda<<<4096, 64>>>
# [00016.333 ms] [1.644725] multiplyCuda<<<4096, 128>>>
# [00015.280 ms] [1.644725] multiplyCuda<<<4096, 256>>>
# [00014.963 ms] [1.644725] multiplyCuda<<<4096, 512>>>
# [00014.966 ms] [1.644725] multiplyCuda<<<4096, 1024>>>
# [00017.452 ms] [1.644725] multiplyCuda<<<8192, 32>>>
# [00016.389 ms] [1.644725] multiplyCuda<<<8192, 64>>>
# [00015.245 ms] [1.644725] multiplyCuda<<<8192, 128>>>
# [00014.981 ms] [1.644725] multiplyCuda<<<8192, 256>>>
# [00014.915 ms] [1.644725] multiplyCuda<<<8192, 512>>>
# [00014.742 ms] [1.644725] multiplyCuda<<<8192, 1024>>>
# [00016.345 ms] [1.644725] multiplyCuda<<<16384, 32>>>
# [00015.224 ms] [1.644725] multiplyCuda<<<16384, 64>>>
# [00014.988 ms] [1.644725] multiplyCuda<<<16384, 128>>>
# [00014.989 ms] [1.644725] multiplyCuda<<<16384, 256>>>
# [00014.764 ms] [1.644725] multiplyCuda<<<16384, 512>>>
# [00014.568 ms] [1.644725] multiplyCuda<<<16384, 1024>>>
# [00015.970 ms] [1.644725] multiplyCuda<<<32768, 32>>>
# [00015.009 ms] [1.644725] multiplyCuda<<<32768, 64>>>
# [00014.963 ms] [1.644725] multiplyCuda<<<32768, 128>>>
# [00014.816 ms] [1.644725] multiplyCuda<<<32768, 256>>>
# [00014.594 ms] [1.644725] multiplyCuda<<<32768, 512>>>
# [00014.947 ms] [1.644725] multiplyCuda<<<32768, 1024>>>
```

[![](https://i.imgur.com/bGUUPot.gif)][sheets]
[![](https://i.imgur.com/IagoPuk.gif)][sheets]
[![](https://i.imgur.com/tCUuW0a.gif)][sheets]
[![](https://i.imgur.com/U6jbPeH.gif)][sheets]

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)

<br>
<br>

[![](https://i.imgur.com/lRwvZLe.png)](https://www.youtube.com/watch?v=vTdodyhhjww)

[charts]: https://photos.app.goo.gl/xorYb1MZSNqxUgNy7
[sheets]: https://docs.google.com/spreadsheets/d/1fWcVNQbANgiNepryktAsIWUHCNiAi-Yf1qQyiLsTJio/edit?usp=sharing
