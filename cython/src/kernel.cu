#include <stdio.h>
#include <cudf/column/column_device_view.cuh>

static const float mm_to_inches = 0.0393701;

__global__ void kernel_tenth_mm_to_inches(cudf::mutable_column_device_view column, int length)
{
    printf("kernel.cu column.data<int64_t>(): %p\n", column.data<int64_t>());
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t col_val = column.element<int64_t>(0);
    printf("Row 0 Value: %lu\n", col_val);
}

__global__ void kernel_mm_to_inches(cudf::mutable_column_device_view column)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // while(gid < length) {
    //     int64_t col_val = column.element<int64_t>(gid); // Returns a reference to the column element value.
    //     col_val = 0;
    //     gid += blockDim.x*gridDim.x;
    // }
}
