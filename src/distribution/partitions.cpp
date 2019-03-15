#include "primitives.h"
#include <algorithm>

namespace ral {
namespace distribution {
namespace partitions {

std::vector<gdf_column_cpp>
makeTableChunk(const std::vector<gdf_column_cpp> &table,
               const gdf_size_type                lower_index,
               const gdf_size_type                upper_index) {
    std::vector<gdf_column_cpp> columnChunks;
    columnChunks.reserve(table.size());

    std::transform(table.cbegin(),
                   table.cend(),
                   std::back_inserter(columnChunks),
                   [lower_index, upper_index](const gdf_column_cpp &column) {
                       gdf_size_type length = upper_index - lower_index;
                       return column.slice(lower_index, length);
                   });

    return columnChunks;
}

std::vector<std::vector<gdf_column_cpp>>
partitionData(const std::vector<gdf_column_cpp> &table,
              const std::vector<gdf_size_type>   pivots) {
    std::vector<std::vector<gdf_column_cpp>> tableChunks;
    tableChunks.reserve(pivots.size() + 1);

    gdf_size_type lower_index = 0;
    for (const gdf_size_type upper_index : pivots) {
        tableChunks.push_back(makeTableChunk(table, lower_index, upper_index));
        lower_index = upper_index;
    }

    return tableChunks;
}

}  // namespace partitions
}  // namespace distribution
}  // namespace ral
