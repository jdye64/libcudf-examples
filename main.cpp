// A very basic and start forward C++ example for using RapidsAI libcudf

#include <iostream>
#include <cudf/table/table.hpp>
#include <cudf/io/types.hpp>
#include <cudf/io/functions.hpp>

#include <cudf/cudf.h>
//#include <cudf/binaryop.hpp>

int main() {
    printf("RapidsAI Simple libcudf Example\n");

    // Make an alias for the io namespace to make more readable
    namespace io = cudf::experimental::io;

    // 1 - Read in sample csv file to libcudf dataframe

    // 1.1 First we define something called "csv_read_arg" this tells 
    // the io_function reader all the parameters its needs to know to 
    // understand how to read and parse the incoming csv data
    // Reference: https://github.com/rapidsai/cudf/blob/cc3855d01d04c7c93f526e5046a75e60a8040501/cpp/include/cudf/io/functions.hpp#L133
    std::string filepath = "/home/jdyer/Development/libcudf-examples/data/weather/2010.csv.gz";
    io::read_csv_args in_args{io::source_info{filepath}};

    // 1.2 Set all of the values for the csv reader arguments
    in_args.header = -1;

    // Give the columns that will be read from the csv file names instead of auto generating random names.
    std::vector<std::string> col_names{ "station_id", "record_date", "observation_type" };
    in_args.names = col_names;

    // 1.3 Perform the actual read which create the cudf::table instance for you. Same thing as Python "DataFrame" instance
    io::table_with_metadata wdf = io::read_csv(in_args);

    // 1.4 This is not required its simply here to illustrate how info about the resulting table instance can be obtained
    setlocale(LC_NUMERIC, "");  // Simply for thousands separator
    printf("Num Columns: %'d Number of Rows: %'d\n", wdf.tbl->num_columns(), wdf.tbl->num_rows());
    
    // Just an example of how to retrieve the table column names
    std::vector<std::string> column_names = wdf.metadata.column_names;

    // 2 - Perform filter and group by aggregation
    
    // 2.1 - Only keep 'PRCP' (Precipitation) events
    //std::unique_ptr<column> result = cudf::experimental::binary_operation();
    

    // 3 - Write results back to parquet file

    // 3.1 Similarly to reading arguments we also need to have "parquet_write_args"
    // Reference:
    std::string output_parquet = "/home/jdyer/Development/libcudf-examples/data/weather/results.parquet";
    io::write_parquet_args out_args{io::sink_info{output_parquet}, wdf.tbl->view()};

    // Actually write the parquet data to file
    io::write_parquet(out_args);

    return 0;
}