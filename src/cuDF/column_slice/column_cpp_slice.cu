#include "cuDF/column_slice/column_cpp_slice.h"
#include "gdf_wrapper/utilities/cudf_utils.h"
#include "gdf_wrapper/utilities/error_utils.h"
#include "Traits/RuntimeTraits.h"

namespace ral {
namespace cudf {

__global__
void column_valid_slice_kernel(std::uint32_t*       valid_dst,
                               const std::uint32_t* valid_src,
                               const gdf_size_type  valid_src_size,
                               const std::uint32_t  rotate_value,
                               const gdf_size_type  start_block,
                               const gdf_size_type  block_length,
                               const std::uint64_t  last_block_mask) {
    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;

    int start = tid + blkid * blksz;
    int step = blksz * gridsz;

    for (int i = start; i < block_length; i += step) {
        std::uint64_t lower_value = valid_src[start_block + i];
        std::uint64_t upper_value = 0UL;
        if (i < (valid_src_size - 1)) {
            upper_value = valid_src[start_block + i + 1];
        }
        std::uint64_t valid_value = lower_value + (upper_value << 32UL);

        std::uint32_t result = (std::uint32_t)(valid_value >> rotate_value);
        if (last_block_mask && (i == (block_length - 1))) {
            result &= last_block_mask;
        }
        valid_dst[i] = result;
    }
}


gdf_error column_cpp_valid_slice(gdf_column_cpp*       output_column,
                                 const gdf_column_cpp* input_column,
                                 const gdf_size_type   start_bit,
                                 const gdf_size_type   bits_length) {
    // guards
    gdf_size_type valid_src_size = ral::traits::get_valid_size(input_column->get_gdf_column());

    GDF_REQUIRE(((start_bit + bits_length) < (valid_src_size * (gdf_size_type)ral::traits::BYTE_SIZE_IN_BITS)), GDF_INVALID_API_CALL);
    if (bits_length == 0) {
        return GDF_SUCCESS;
    }

    // generate params
    const int word_width = 32;
    const std::uint32_t rotate_value = start_bit % word_width;
    gdf_size_type start_block = start_bit / word_width;
    gdf_size_type block_length = (bits_length / word_width) + ((bits_length % word_width) ? 1 : 0);
    std::uint64_t last_block_mask = (1UL << (bits_length % word_width)) - 1UL;

    // calculate grid and block size
    int block_size;
    int min_grid_size;
    int grid_size;

    CUDA_TRY(
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, column_valid_slice_kernel)
    );
    int required_grid_size = (block_length + block_size - 1) / block_size;
    grid_size = std::min(required_grid_size, min_grid_size);

    // execute kernel
    column_valid_slice_kernel<<< grid_size, block_size >>>((std::uint32_t*) output_column->valid(),
                                                           (const std::uint32_t*) input_column->valid(),
                                                           valid_src_size,
                                                           rotate_value,
                                                           start_block,
                                                           block_length,
                                                           last_block_mask);
    cudaDeviceSynchronize();
    CUDA_TRY(cudaPeekAtLastError());

    return GDF_SUCCESS;
}

} // namespace cudf
} // namespace ral
