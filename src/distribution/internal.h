#ifndef BLAZINGDB_RAL_DISTRIBUTION_INTERNAL_H_
#define BLAZINGDB_RAL_DISTRIBUTION_INTERNAL_H_

#include <GDFColumn.cuh>

namespace internal {

gdf_column_cpp slice(const gdf_column_cpp &col,
                     const gdf_size_type   start,
                     const gdf_size_type   length);

}  // namespace internal

#endif
