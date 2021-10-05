#pragma once
#include "_main.hxx"




struct MultiplyOptions {
  int repeat;
  int blockSize;
  int threadDuty;

  MultiplyOptions(int repeat=1, int blockSize=BLOCK_LIMIT, int threadDuty=1) :
  repeat(repeat), blockSize(blockSize), threadDuty(threadDuty) {}
};
