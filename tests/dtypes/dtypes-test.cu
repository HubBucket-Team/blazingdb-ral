#include <type_traits>

#include <gtest/gtest.h>

#include <CalciteInterpreter.h>
#include <GDFColumn.cuh>
#include <gdf/gdf.h>

template <class T>
class DTypesTest : public ::testing::Test {
protected:
  void Check(gdf_column_cpp out_col, T *host_output) {
    T *device_output;
    device_output = new T[out_col.size()];
    cudaMemcpy(device_output,
               out_col.data(),
               out_col.size() * sizeof(T),
               cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < out_col.size(); i++) {
      ASSERT_TRUE(host_output[i] == device_output[i]);
    }
  }
};

template <class U>
struct DTypeTraits {};

#define DTYPE_FACTORY(U, D)                                                    \
  template <>                                                                  \
  struct DTypeTraits<U> {                                                      \
    static constexpr gdf_dtype dtype = GDF_##D;                                \
  }

DTYPE_FACTORY(std::int8_t, INT8);
DTYPE_FACTORY(std::int16_t, INT16);
DTYPE_FACTORY(std::int32_t, INT32);
DTYPE_FACTORY(std::int64_t, INT64);
DTYPE_FACTORY(std::uint8_t, UINT8);
DTYPE_FACTORY(std::uint16_t, UINT16);
DTYPE_FACTORY(std::uint32_t, UINT32);
DTYPE_FACTORY(std::uint64_t, UINT64);
DTYPE_FACTORY(float, FLOAT32);
DTYPE_FACTORY(double, FLOAT64);

#undef DTYPE_FACTORY

using DTypesTestTypes = ::testing::Types<std::int8_t,
                                         std::int16_t,
                                         std::int32_t,
                                         std::int64_t,
                                         std::uint8_t,
                                         std::uint16_t,
                                         std::uint32_t,
                                         std::uint64_t,
                                         float,
                                         double>;
TYPED_TEST_CASE(DTypesTest, DTypesTestTypes);

template <class T = void>
class floating : public std::false_type {};
template <>
class floating<float> : public std::true_type {};
template <>
class floating<double> : public std::true_type {};

TYPED_TEST(DTypesTest, withGdfDType) {
  const std::size_t num_values = 100;

  TypeParam *input1 = new TypeParam[num_values];
  TypeParam *input2 = new TypeParam[num_values];
  TypeParam *input3 = new TypeParam[num_values];

  for (int i = 0; i < num_values; i++) {
    if (i % 2 == 0) {
      input1[i] = 1;
    } else {
      input1[i] =
        floating<TypeParam>::value ? static_cast<TypeParam>(i) / 1000 : i;
    }
    input2[i] =
      floating<TypeParam>::value ? static_cast<TypeParam>(i) / 100000 : i;
    input3[i] = 1;
  }

  std::vector<gdf_column_cpp> inputs;
  inputs.resize(3);
  inputs[0].create_gdf_column(DTypeTraits<TypeParam>::dtype,
                              num_values,
                              (void *) input1,
                              sizeof(TypeParam));
  inputs[1].create_gdf_column(DTypeTraits<TypeParam>::dtype,
                              num_values,
                              (void *) input2,
                              sizeof(TypeParam));
  inputs[2].create_gdf_column(DTypeTraits<TypeParam>::dtype,
                              num_values,
                              (void *) input3,
                              sizeof(TypeParam));

  std::vector<std::vector<gdf_column_cpp> > input_tables;
  std::vector<std::string>               table_names  = {"hr.emps", "hr.sales"};
  std::vector<std::vector<std::string> > column_names = {{"x", "y", "z"},
                                                         {"a", "b", "x"}};
  input_tables.push_back(inputs);
  input_tables.push_back(inputs);
  std::vector<gdf_column_cpp> outputs;
  {
    std::string query = "\
LogicalProject(S=[-($0, $1)])\n\
  EnumerableTableScan(table=[[hr, emps]])";

    gdf_error err =
      evaluate_query(input_tables, table_names, column_names, query, outputs);
    EXPECT_TRUE(err == GDF_SUCCESS);
    EXPECT_TRUE(outputs.size() == 1);

    TypeParam *host_output = new TypeParam[num_values];
    for (std::size_t i = 0; i < num_values; i++) {
      host_output[i] = input1[i] - input2[i];
    }

    this->Check(outputs[0], host_output);
  }
}
