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

#include <cudf/stream_compaction.hpp>
#include <cudf/groupby.hpp>
#include <cudf/aggregation.hpp>


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
    in_args.names = std::vector<std::string> { "station_id", "date", "type", "val", "m_flag", "q_flag", "s_flag", "obs_time"};
    in_args.use_cols_names = std::vector<std::string> { "station_id", "date", "type", "val" };

    // 1.3 Perform the actual read which create the cudf::table instance for you. Same thing as Python "DataFrame" instance
    io::table_with_metadata wdf = io::read_csv(in_args);

    printf("# Total Rows %d\n", wdf.tbl->num_rows());

    // 2 - Weather data Logic section
    
    // 2.1 - Only keep 'PRCP' (Precipitation) events
    std::unique_ptr<cudf::column> prcp_cols = cudf::experimental::binary_operation(cudf::string_scalar("PRCP"), 
                                                       wdf.tbl->view().column(2), // "val" column
                                                       cudf::experimental::binary_operator::EQUAL,
                                                       cudf::data_type(cudf::experimental::type_to_id<bool>()));

    std::unique_ptr<cudf::experimental::table> prcp_tbl = cudf::experimental::apply_boolean_mask(wdf.tbl->view(), prcp_cols->view());
    printf("# PRCP Rows %d\n", prcp_tbl->num_rows());

    // 2.2 - Group by Station ID
    namespace gb = cudf::experimental::groupby;
    gb::groupby station_gb(prcp_tbl->view());

    // 2.2.1 - Create the Aggregation requests that should be used by the groupby
    std::vector<gb::aggregation_request> agg_requests;

    // Create the SUM() aggregation requests for rainfall totals by station id
    gb::aggregation_request ar_totals;
    ar_totals.values = std::move(prcp_tbl->get_column(3).view());
    
    std::vector<std::unique_ptr<cudf::experimental::aggregation>> aggs;
    aggs.push_back(std::move(cudf::experimental::make_sum_aggregation()));
    ar_totals.aggregations = std::move(aggs);

    agg_requests.push_back(std::move(ar_totals));
    
    std::pair<std::unique_ptr<cudf::experimental::table>, std::vector<gb::aggregation_result>> gb_results = station_gb.aggregate(agg_requests);

    // 3 - Write results back to parquet file

    // 3.1 Similarly to reading arguments we also need to have "parquet_write_args"
    printf("# GroupBy Rows %d\n", gb_results.first->num_rows());

    std::string output_parquet = "/home/jdyer/Development/libcudf-examples/data/weather/results_final.parquet";

    // Create the final table with the results
    std::vector<std::unique_ptr<cudf::column>> cols = gb_results.first->release();
    cols.push_back(std::move(gb_results.second.at(0).results.at(0)));
    
    cudf::experimental::table prcp_events = cudf::experimental::table(std::move(cols));

    io::write_parquet_args out_args{io::sink_info{output_parquet}, prcp_events.view()};
    io::write_parquet(out_args);

    return 0;
}