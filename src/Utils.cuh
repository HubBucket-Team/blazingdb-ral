#ifndef UTILS_CUH_
#define UTILS_CUH_

#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <iostream>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

void print_column(gdf_column * column);

void free_gdf_column(gdf_column * column);

void create_gdf_column(gdf_column * column, gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);

void realloc_gdf_column(gdf_column * column, size_t size, size_t width);

typedef gdf_column* key_type;

// Type for a unique_ptr to a gdf_column with a custom deleter
// Custom deleter is defined at construction
using gdf_col_pointer = typename std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;
using gdf_col_shared_pointer = typename std::shared_ptr<gdf_column>;

gdf_col_pointer create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);
gdf_col_shared_pointer create_gdf_shared_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);

/*class gdf_column_cpp
{
    private:
        gdf_column column;

    public:
        gdf_column* get_gdf_column();

        gdf_column_cpp(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);

        gdf_column_cpp(const gdf_column_cpp& col);

        void create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);

        gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid, gdf_size_type size, gdf_dtype dtype);

        ~gdf_column_cpp();
};*/

#endif /* UTILS_CUH_ */
