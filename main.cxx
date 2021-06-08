#include <vector>
#include <cstdio>
#include "src/main.hxx"

using namespace std;




void runMultiply(int N) {
  int repeat = 5;
  vector<double> a1(N), a2(N);
  vector<double> x(N), y(N);
  for (int i=0; i<N; i++) {
    x[i] = 1.0/(i+1);
    y[i] = 1.0/(i+1);
  }

  // Find x*y using a single thread.
  float t1 = multiplySeq(a1, x, y, {repeat});
  printf("[%09.3f ms; %.0e elems.] [%f] multiplySeq\n", t1, (double) N, sum(a1));

  // Find x*y accelerated using CUDA.
  for (int grid=1024; grid<=GRID_LIMIT; grid*=2) {
    for (int block=32; block<=BLOCK_LIMIT; block*=2) {
      float t2 = multiplyCuda(a2, x, y, {repeat, grid, block});
      printf("[%09.3f ms; %.0e elems.] [%f] multiplyCuda<<<%d, %d>>>\n", t1, (double) N, sum(a1), grid, block);
    }
  }
}


int main(int argc, char **argv) {
  for (int n=10000; n<=1000000000; n*=10) {
    runMultiply(n);
    printf("\n");
  }
  return 0;
}
