#include <algorithm>
#include <gdf_wrapper/gdf_wrapper.cuh>
#include "cuDF/concat/concat_merge.h"

namespace cudf {
namespace concat {

    gdf_error concat_merge(std::vector<blazing_frame>& input_frames,
                           blazing_frame& output_frame,
                           bool destroy_inputs) {

        // verify whether there is data to concatenate
        if (input_frames.size() == 0) {
            return GDF_DATASET_EMPTY;
        }

        // use std::vector<std::vector<gdf_column_cpp>> as input
        auto& input_frame_vectors = input_frames[0].get_columns();

        // reserve output data
        // use std::vector<gdf_column_cpp> as output
        output_frame.get_columns().resize(1);
        auto& output_frame_vector = output_frame.get_columns()[0];
        output_frame_vector.reserve(input_frame_vectors.size());

        // execute concatenation
        for (std::size_t i = 0; i < input_frame_vectors.size(); ++i) {
            // verify if dataset is empty
            if (input_frame_vectors[i].size() == 0) {
                continue;
            }

            // get first input
            auto& first_input = input_frame_vectors[i][0];

            // get the size of the dtype
            int width;
            auto error = get_column_byte_width(first_input.get_gdf_column(), &width);
            if (error != GDF_SUCCESS) {
                output_frame.clear();
                return error;
            }

            // get total size of the input data.
            // verify whether all the gdf_column_cpp have the same dtype.
            std::size_t total_size_concat = 0;
            gdf_dtype dtype_concat = first_input.dtype();
            for (std::size_t k = 0; k < input_frame_vectors[i].size(); ++k) {
                auto& column = input_frame_vectors[i][k];
                if (dtype_concat != column.dtype()) {
                    output_frame.clear();
                    return GDF_DTYPE_MISMATCH;
                }
                total_size_concat += column.size();
            }

            // create output gdf_column
            gdf_column_cpp output_column;
            output_column.create_gdf_column(first_input.dtype(),
                                            total_size_concat,
                                            nullptr,
                                            width,
                                            "concat-" + first_input.name());

            // create a container with all of the gdf_column inputs
            std::vector<gdf_column*> columns;
            columns.reserve(input_frame_vectors[i].size());
            for (std::size_t k = 0; k < input_frame_vectors[i].size(); ++k) {
                columns.emplace_back(input_frame_vectors[i][k].get_gdf_column());
            }

            // perform concat function
            error = gdf_column_concat(output_column.get_gdf_column(), columns.data(), columns.size());
            if (error != GDF_SUCCESS) {
                output_frame.clear();
                return error;
            }

            // add concat column to the output
            output_frame_vector.emplace_back(output_column);
        }

        // clear input
        if (destroy_inputs) {
            input_frames.clear();
        }

        return GDF_SUCCESS;
    }

} // namespace concat
} // namespace cudf
