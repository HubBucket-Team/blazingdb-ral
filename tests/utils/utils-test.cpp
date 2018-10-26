#include <gtest/gtest.h>

#include "gdf/library/table.h"
#include "gdf/library/table_group.h"

using namespace gdf::library;

TEST(UtilsTest, TableBuilder) {
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
  t.print(std::cout);
}

TEST(UtilsTest, FrameFromTableGroup) {
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

  g[0].print(std::cout);
  g[1].print(std::cout);

  BlazingFrame frame = g.ToBlazingFrame();

  auto hostVector = HostVectorFrom<GDF_UINT64>(frame[1][1]);

  for (std::size_t i = 0; i < 20; i++) { EXPECT_EQ(i * 10000, hostVector[i]); }
}

TEST(UtilsTest, TableFromLiterals) {
  auto t = LiteralTableBuilder{"emps",
                               {
                                 {"x", Literals<GDF_FLOAT64>{1, 3, 5, 7, 9}},
                                 {"y", Literals<GDF_INT64>{0, 2, 4, 6, 8}},
                               }}
             .Build();

  for (std::size_t i = 0; i < 5; i++) {
    EXPECT_EQ(2 * i, t[1][i].get<GDF_INT64>());
  }

  for (std::size_t i = 0; i < 5; i++) {
    EXPECT_EQ(2 * i + 1.0, t[0][i].get<GDF_FLOAT64>());
  }
}
