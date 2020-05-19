#include <stdio.h>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>

static const float mm_to_inches = 0.0393701;

__global__ void kernel_tenth_mm_to_inches(cudf::mutable_table_device_view table, int length)
{
    printf("Howdy from here\n");
    printf("Number Columns: %lu, Number Rows: %lu\n", table.num_columns(), table.num_rows());
    //printf("kernel.cu table.data<int64_t>(): %p\n", table.data<int64_t>());
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
//     int64_t col_val = column.element<int64_t>(0);
//     printf("Row 0 Value: %lu\n", col_val);
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
