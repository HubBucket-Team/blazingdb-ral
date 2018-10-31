#include <CalciteInterpreter.h>

#include <gdf/library/table_group.h>

#include <gtest/gtest.h>

class WhereTest : public testing::Test {};

using gdf::library::Literals;
using gdf::library::LiteralTableBuilder;
using gdf::library::TableGroup;

TEST_F(WhereTest, CompareLiteralWhere) {
  auto table = LiteralTableBuilder{
    "main.holas",
    {
      {
        "swings",
        Literals<GDF_INT64>{1, 2, 3, 4, 5, 6},
      },
      {
        "tractions",
        Literals<GDF_INT64>{1, 0, 1, 0, 1, 0},
      },
    }}.Build();
  auto group = TableGroup{table};

  std::string plan = "LogicalProject(swings=[$0])\n"
                     "  LogicalFilter(condition=[=($1, 1)])\n"
                     "    EnumerableTableScan(table=[[main, holas]])";

  std::vector<gdf_column_cpp> output;

  gdf_error status = evaluate_query(group.ToBlazingFrame(),
                                    group.table_names(),
                                    group.column_names(),
                                    plan,
                                    output);
  EXPECT_EQ(GDF_SUCCESS, status);

  auto result =
    gdf::library::GdfColumnCppsTableBuilder{"ResultTable", output}.Build();

  auto expected = LiteralTableBuilder{
    "ResultTable",
    {
      {"", Literals<GDF_INT64>{1, 3, 5, 0, 0, 0}},
    }}.Build();

  EXPECT_EQ(expected, result);
}
