#include <CalciteInterpreter.h>
#include <gdf/library/api.h>

#include <gtest/gtest.h>

TEST(WithoutGroupByTest, Init) {
    using namespace gdf::library;

    // select count(*) from main.nation
    std::string logicalPlan = "LogicalAggregate(group=[{}], EXPR$0=[COUNT()])\n"
                              "  LogicalProject($f0=[0])\n"
                              "    EnumerableTableScan(table=[[main, nation]])";

    auto tableGroup = LiteralTableGroupBuilder{
      {"main.nation",
       {
         {"col1", Literals<GDF_INT64>{1, 3, 5, 7, 9}},
         {"col2", Literals<GDF_INT64>{0, 2, 4, 6, 8}},
       }}}.Build();

    auto inputTables = tableGroup.ToBlazingFrame();
    auto tableNames  = tableGroup.table_names();
    auto columnNames = tableGroup.column_names();

    std::vector<gdf_column_cpp> outputs;

    gdf_error gdfError = evaluate_query(
      inputTables, tableNames, columnNames, logicalPlan, outputs);

    EXPECT_EQ(GDF_SUCCESS, gdfError);
    EXPECT_EQ(1, outputs.size());

    auto outputTable =
      GdfColumnCppsTableBuilder{"outputTable", outputs}.Build();

    EXPECT_EQ(5, outputTable[0][0].get<GDF_INT64>());
}
