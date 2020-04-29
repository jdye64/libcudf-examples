// A very basic and start forward C++ example for using RapidsAI libcudf

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

    // 2 - Weather data Logic section
    
    // 2.1 - Filter for 'PRCP' (Precipitation) events using a boolean_mask/binaryop
    std::unique_ptr<cudf::column> prcp_cols = cudf::experimental::binary_operation(cudf::string_scalar("PRCP"), 
                                                       wdf.tbl->view().column(2), // "type" column
                                                       cudf::experimental::binary_operator::EQUAL,
                                                       cudf::data_type(cudf::experimental::type_to_id<bool>()));

    std::unique_ptr<cudf::experimental::table> prcp_tbl = cudf::experimental::apply_boolean_mask(wdf.tbl->view(), prcp_cols->view());
    printf("# PRCP Weather Rows %d\n", prcp_tbl->num_rows());
    std::vector<std::unique_ptr<cudf::column>> all_p_cols = prcp_tbl->release();

    // 2.2 - Group by Station ID
    namespace gb = cudf::experimental::groupby;

    std::vector<std::unique_ptr<cudf::column>> key_p_cols;
    key_p_cols.push_back(std::move(all_p_cols.at(0)));      // Place the "station_id" column in desired groupby 'keys' column list

    cudf::experimental::table prcp_keys = cudf::experimental::table(std::move(key_p_cols)); // table with only "station_id" column

    gb::groupby station_gb(prcp_keys.view());   // Create groupby object with "station_id" only column table.

    // 2.2.1 - Create Aggregation requests for the groupby object
    std::vector<gb::aggregation_request> agg_requests;

    // Create the SUM() aggregation requests for rainfall totals by station id
    gb::aggregation_request ar_totals;
    ar_totals.values = std::move(all_p_cols.at(3)->view());
    
    std::vector<std::unique_ptr<cudf::experimental::aggregation>> aggs;
    aggs.push_back(std::move(cudf::experimental::make_sum_aggregation()));
    ar_totals.aggregations = std::move(aggs);

    agg_requests.push_back(std::move(ar_totals));
    
    std::pair<std::unique_ptr<cudf::experimental::table>, std::vector<gb::aggregation_result>> gb_results = station_gb.aggregate(agg_requests);

    // 3 - Write results back to parquet file

    // 3.1 Similarly to reading arguments we also need to have "parquet_write_args"

    // Create the final table with the results
    std::vector<std::unique_ptr<cudf::column>> cols = gb_results.first->release();
    cols.push_back(std::move(gb_results.second.at(0).results.at(0)));
    
    cudf::experimental::table prcp_events = cudf::experimental::table(std::move(cols)); // Resulting table with original columns and SUM column all in place

    io::write_parquet_args out_args{io::sink_info{output_path}, prcp_events.view()};
    io::write_parquet(out_args);

    return 0;
}
