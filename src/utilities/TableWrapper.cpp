#include "utilities/TableWrapper.h"
#include "GDFColumn.cuh"

namespace ral {
namespace utilities {

TableWrapper::TableWrapper(const std::vector<gdf_column_cpp>& columns) {
    size_ = columns.size();
    columns_ = new gdf_column*[size_];
    for (gdf_size_type i = 0; i < size_; ++i) {
        columns_[i] = columns[i].get_gdf_column();
    }
}

TableWrapper::~TableWrapper() {
    if (columns_ != nullptr) {
        delete[] columns_;
        columns_ = nullptr;
    }
}

gdf_column** TableWrapper::getColumns() {
    return columns_;
}

gdf_size_type TableWrapper::getQuantity() {
    return size_;
}

} // namespace utilities
} // namespace ral
