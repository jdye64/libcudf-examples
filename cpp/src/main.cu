#include <stdio.h>

#include <iostream>
#include <cudf/table/table.hpp>
#include <cudf/io/types.hpp>
#include <cudf/io/functions.hpp>

#include <cudf/binaryop.hpp>

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>

#include <cudf/stream_compaction.hpp>
#include <cudf/groupby.hpp>
#include <cudf/aggregation.hpp>

#include <rmm/device_buffer.hpp>

__global__ void cudfUDF(void *data, int *dev_size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //if (i < n) y[i] = a*x[i] + y[i];
  *dev_size = sizeof(data);
}

  int main(int argc, char** argv) {
    printf("RapidsAI Weather libcudf Example\n");

    if (argc != 3) {
        printf("You must specify a weather CSV input file and and output location for resulting parquet.\n");
        return 1;
    }

    std::string input_csv_file = argv[1];
    std::string output_path = argv[2];

    printf("Reading weather file: %s and writing results to location: %s\n", input_csv_file.c_str(), output_path.c_str());

    // Make an alias for the io namespace to make more readable
    namespace io = cudf::experimental::io;

    // 1 - Read in sample csv file to libcudf dataframe

    // 1.1 First we define something called "csv_read_arg" this tells 
    // the io_function reader all the parameters its needs to know to 
    // understand how to read and parse the incoming csv data
    // Reference: https://github.com/rapidsai/cudf/blob/cc3855d01d04c7c93f526e5046a75e60a8040501/cpp/include/cudf/io/functions.hpp#L133
    io::read_csv_args in_args{io::source_info{input_csv_file}};

    // 1.2 Set all of the values for the csv reader arguments
    in_args.header = -1;
    in_args.names = std::vector<std::string> { "station_id", "date", "type", "val", "m_flag", "q_flag", "s_flag", "obs_time"};
    in_args.use_cols_names = std::vector<std::string> { "station_id", "date", "type", "val" };

    // 1.3 Perform the actual read which create the cudf::table instance for you. Same thing as Python "DataFrame" instance
    io::table_with_metadata wdf = io::read_csv(in_args);
    printf("# Total Weather Rows %d\n", wdf.tbl->num_rows());

    int *size, *d_size;
    size = (int*)malloc(sizeof(int));
    cudaMalloc(&d_size, sizeof(int));

    cudaMemcpy(d_size, size, sizeof(int), cudaMemcpyHostToDevice);

    // Perform cudfUDF on weather data
    std::unique_ptr<rmm::device_buffer> b = wdf.tbl->get_column(0).release().data;
    cudfUDF<<<1, 1>>>(b->data(), d_size);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cudaMemcpy(size, d_size, sizeof(int), cudaMemcpyDeviceToHost);

    printf("# of CUDA rows is: %d\n", *size);

    return 0;
}