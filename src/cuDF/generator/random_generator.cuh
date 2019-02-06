#pragma once

#include <ctime>
#include <vector>
#include <thrust/random.h>
#include <thrust/device_vector.h>

namespace cudf {
namespace generator {

    template <typename T>
    struct UniformRandomGenerator {
        UniformRandomGenerator(T min_value, T max_value, unsigned int time_value)
        : min_value{min_value}, max_value{max_value}, time_value{time_value}
        { }

        __host__ __device__
        unsigned int hash(unsigned int index, unsigned int min_value, unsigned int max_value) {
            return ((time_value * 96377 + (index + 1) * 95731 + (min_value + 1) * 96517 + (max_value + 1) * 99991) % 1073676287);
        }

        __host__ __device__
        T operator()(std::size_t index) {
            thrust::default_random_engine random_engine(hash(index, min_value, max_value));
            thrust::random::uniform_int_distribution<T> distribution(min_value, max_value - 1);
            random_engine.discard(index);
            return distribution(random_engine);
        }

        const T min_value;
        const T max_value;
        const unsigned int time_value;
    };


    template <typename T>
    class RandomVectorGenerator {
    public:
        RandomVectorGenerator(T min_value, T max_value)
        : min_value{min_value}, max_value{max_value}
        { }

    public:
        std::vector<T> operator()(std::size_t size) {
            thrust::device_vector<T> data(size);

            thrust::transform(thrust::counting_iterator<T>(0),
                              thrust::counting_iterator<T>(size),
                              data.begin(),
                              UniformRandomGenerator<T>(min_value, max_value, time(NULL)));

            std::vector<T> result(size);
            thrust::copy(data.begin(), data.end(), result.begin());

            return result;
        }

    private:
        const T min_value;
        const T max_value;
    };

} // namespace generator
} // namespace cudf
