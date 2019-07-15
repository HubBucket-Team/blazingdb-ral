#include "Utils.cuh"
#include "cuDF/Allocator.h"

void gdf_sequence(int32_t* data, size_t size, int32_t init_val){
  auto d_ptr = thrust::device_pointer_cast(data);
  thrust::sequence(d_ptr, d_ptr + size, init_val);
}

void gdf_sequence(int32_t* data, size_t size, int32_t init_val, int32_t step){
  auto d_ptr = thrust::device_pointer_cast(data);
  thrust::sequence(d_ptr, d_ptr + size, init_val, step);
}
