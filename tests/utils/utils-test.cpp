#include <gtest/gtest.h>

#include "utils.h"

TEST(UtilsTest, InitData) {
  using namespace ral::test::utils;

  Table t =
    TableBuilder{
      "emps",
      {
        {"x", [](const std::size_t) -> DType<GDF_FLOAT64> { return .1; }},
        {"y", [](const std::size_t i) -> DType<GDF_UINT64> { return i; }},
      }}
      .Build(10);

  for (std::size_t i = 0; i < 10; i++) {
    EXPECT_EQ(i, t[1].get<GDF_UINT64>(i));
  }

  for (std::size_t i = 0; i < 10; i++) {
    EXPECT_EQ(.1, t[0].get<GDF_FLOAT64>(i));
  }
}
