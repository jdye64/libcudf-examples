// A very basic and start forward C++ example for using RapidsAI libcudf

#include <iostream>
#include <cudf/table/table.hpp>
#include <cudf/io/types.hpp>
#include <cudf/io/functions.hpp>

int main() {
    printf("RapidsAI Simple libcudf Example\n");

    // Make an alias for the io namespace to make more readable
    namespace io = cudf::experimental::io;

    // 1 - Read in sample csv file to libcudf dataframe

    // 1.1 First we define something called "csv_read_arg" this tells 
    // the io_function reader all the parameters its needs to know to 
    // understand how to read and parse the incoming csv data
    // Reference: https://github.com/rapidsai/cudf/blob/cc3855d01d04c7c93f526e5046a75e60a8040501/cpp/include/cudf/io/functions.hpp#L133
    std::string filepath = "/home/jdyer/Development/libcudf-examples/data/weather/2000.csv.gz";
    io::read_csv_args in_args{io::source_info{filepath}};

    // 1.2 Set all of the values for the csv reader arguments
    //in_args.header = -1;

    // 1.3 Perform the actual read which create the cudf::table instance for you. Same thing as Python "DataFrame" instance
    //io::table_with_metadata gdf = io::read_csv(args);
    auto result = io::read_csv(in_args);

    // 2 - Perform filter and group by aggregation
    

    // 3 - Write results back to csv file

    // 3.1 Similarly to reading arguments we also need to have "csv_write_args"
    // Reference: 

    return 0;
}