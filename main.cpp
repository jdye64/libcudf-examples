// A very basic and start forward C++ example for using RapidsAI libcudf

#include <iostream>
#include <cudf/table/table.hpp>
#include <cudf/io/types.hpp>
#include <cudf/io/functions.hpp>

#include <cudf/cudf.h>
#include <cudf/binaryop.hpp>

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/copying.hpp>

#include <cudf/detail/copy_if.cuh>


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

    // // 1.4 This is not required its simply here to illustrate how info about the resulting table instance can be obtained
    // setlocale(LC_NUMERIC, "");  // Simply for thousands separator
    // printf("Num Columns: %'d Number of Rows: %'d\n", wdf.tbl->num_columns(), wdf.tbl->num_rows());
    
    // // Just an example of how to retrieve the table column names
    // std::vector<std::string> column_names = wdf.metadata.column_names;

    // 2 - Weather data Logic section
    
    // 2.1 - Only keep 'PRCP' (Precipitation) events
    using TypeOut = bool;
    using TypeLhs = std::string;
    using TypeRhs = std::string;

    // cudf::experimental::binary_operator eq_precip_binop = cudf::experimental::binary_operator::EQUAL;
    //cudf::string_scalar prcp_scalar("PRCP");

    // auto result = cudf::experimental::binary_operation(prcp_scalar, 
    //                                                    wdf.tbl->view().column(2), // "observation_type" column
    //                                                    cudf::experimental::binary_operator::EQUAL,
    //                                                    cudf::data_type(cudf::experimental::type_to_id<TypeOut>()));

    //using EQUAL = cudf::library::operation::Equal<TypeOut, TypeLhs, TypeRhs>;

    // Create a Filter/Functor to filter only for the desired "PRCP" weather events. Ensure __device__ is present to run in CUDA!
    struct prcp_evt_functor {
        bool operator()(cudf::size_type i) const { return true; }
    };


    std::unique_ptr<cudf::experimental::table> prcp_tbl = cudf::experimental::detail::copy_if(wdf.tbl->view(), prcp_evt_functor);

    // 3 - Write results back to parquet file

    // 3.1 Similarly to reading arguments we also need to have "parquet_write_args"
    // Reference:
    std::string output_parquet = "/home/jdyer/Development/libcudf-examples/data/weather/results.parquet";
    //io::write_parquet_args out_args{io::sink_info{output_parquet}, wdf.tbl->view()};
    //std::make_uni<cudf::experimental::table> out_table = 
    io::write_parquet_args out_args{io::sink_info{output_parquet}, prcp_tbl->view()};

    // Actually write the parquet data to file
    io::write_parquet(out_args);

    return 0;
}