#ifndef CUDF_NVCATEGORY_GATHER_HPP_
#define CUDF_NVCATEGORY_GATHER_HPP_

#include <nvstrings/NVCategory.h>
#include <string/nvcategory_util.hpp>

namespace ral {

// TODO(cudf): We need to check output size because nvcategory_gather
// doesn't create a empty NVCategory for output gdf column, so it could
// produce crashes
inline gdf_error
safe_nvcategory_gather_for_string_category(gdf_column * column,
                                           NVCategory * nv_category) {
    gdf_error error = nvcategory_gather(column, nv_category);

    if (0 == column->size) {
        column->dtype_info.category = NVCategory::create_from_array(nullptr, 0);
    }

    return error;
}

inline gdf_error safe_nvcategory_gather_for_string_category(gdf_column * column,
                                                            void * category) {
    return safe_nvcategory_gather_for_string_category(
        column, static_cast<NVCategory *>(category));
}

}  // namespace ral

#endif
