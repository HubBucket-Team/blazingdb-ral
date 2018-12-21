#include <CalciteInterpreter.h>
#include <gdf/library/api.h>

#include <gtest/gtest.h>

class Item {
public:
    const std::string  logicalPlan;
    const std::int64_t expectedOutputLength;
    const std::int64_t expectedCount;
};

class WithoutGroupByTest : public testing::TestWithParam<Item> {};

TEST_P(WithoutGroupByTest, RunQuery) {
    using namespace gdf::library;

    auto tableGroup = LiteralTableGroupBuilder{
      {"main.nation",
       {
         {"n_nationkey", Literals<GDF_INT64>{1, 3, 5, 1, 3, 7, 9}},
         {"n_regionkey", Literals<GDF_INT64>{0, 2, 4, 6, 8, 0, 2}},
       }}}.Build();

    auto inputTables = tableGroup.ToBlazingFrame();
    auto tableNames  = tableGroup.table_names();
    auto columnNames = tableGroup.column_names();

    std::uint8_t mask = 0b11111000;
    for (auto gdfColumnCpp : inputTables[0]) {
        gdf_column *gdfColumn = gdfColumnCpp.get_gdf_column();
        cudaMemcpy(gdfColumn->valid, &mask, 1, cudaMemcpyHostToDevice);
        gdfColumn->null_count = 5;
    }

    std::vector<gdf_column_cpp> outputs;

    gdf_error gdfError = evaluate_query(
      inputTables, tableNames, columnNames, GetParam().logicalPlan, outputs);

    EXPECT_EQ(GDF_SUCCESS, gdfError);
    EXPECT_EQ(GetParam().expectedOutputLength, outputs.size());

    auto outputTable =
      GdfColumnCppsTableBuilder{"outputTable", outputs}.Build();

    EXPECT_EQ(GetParam().expectedCount, outputTable[0][0].get<GDF_INT64>());
}

INSTANTIATE_TEST_CASE_P(
  QueriesWithCount,
  WithoutGroupByTest,
  testing::ValuesIn({
    Item{// select count(*) from main.nation
         "LogicalAggregate(group=[{}], EXPR$0=[COUNT()])\n"
         "  LogicalProject($f0=[0])\n"
         "    EnumerableTableScan(table=[[main, nation]])",
         1,
         7},
    Item{// select count(n_nationkey) from main.nation
         "LogicalAggregate(group=[{}], EXPR$0=[COUNT()])\n"
         "  LogicalProject(n_nationkey=[$0])\n"
         "    EnumerableTableScan(table=[[main, nation]])",
         1,
         5},
  }));
