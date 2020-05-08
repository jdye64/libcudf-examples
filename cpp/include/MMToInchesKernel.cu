#include <stdio.h>
#include <iostream>

__global__ void cudfUDF(void *data, int *dev_size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //if (i < n) y[i] = a*x[i] + y[i];
  *dev_size = sizeof(data);
}
