#pragma once

#include <vector>
#include "DataFrame.h"

namespace cudf {
namespace concat {

    gdf_error concat_merge(std::vector<blazing_frame>& input_frames,
                           blazing_frame& output_frame,
                           bool destroy_inputs = false);

} // namespace concat
} // namespace cudf
