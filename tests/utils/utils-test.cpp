#include <gtest/gtest.h>

#include "utils.h"

TEST(UtilsTest, InitData) {
  using namespace ral::test::utils;

  Table t =
    TableBuilder{
      "emps",
      {
        {"x", [](Index i) -> DType<GDF_FLOAT64> { return i / 10.0; }},
        {"y", [](Index i) -> DType<GDF_UINT64> { return i * 1000; }},
      }}
      .Build(10);

  for (std::size_t i = 0; i < 10; i++) {
    EXPECT_EQ(i * 1000, t[1][i].get<GDF_UINT64>());
  }

  for (std::size_t i = 0; i < 10; i++) {
    EXPECT_EQ(i / 10.0, t[0][i].get<GDF_FLOAT64>());
  }

  auto g =
    TableGroupBuilder{
      {"emps",
       {
         {"x", [](Index i) -> DType<GDF_FLOAT64> { return i / 10.0; }},
         {"y", [](Index i) -> DType<GDF_UINT64> { return i * 1000; }},
       }},
      {"emps",
       {
         {"x", [](Index i) -> DType<GDF_FLOAT64> { return i / 100.0; }},
         {"y", [](Index i) -> DType<GDF_UINT64> { return i * 10000; }},
       }}}
      .Build({10, 20});

  BlazingFrame frame = g.ToBlazingFrame();

  auto hostVector = HostVectorFrom<GDF_UINT64>(frame[1][1]);

  for (std::size_t i = 0; i < 20; i++) {
    EXPECT_EQ(i * 10000, hostVector[i]);
  }
}
