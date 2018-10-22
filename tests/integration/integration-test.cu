#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

#include "gtest/gtest.h"
#include <CalciteExpressionParsing.h>
#include <CalciteInterpreter.h>
#include <StringUtil.h>
#include <DataFrame.h>
#include <GDFColumn.cuh>
#include <GDFCounter.cuh>
#include <Utils.cuh>

#include <gdf/gdf.h>
//ToDo Script for generate and/or update this header while cmaking
#include "generated.h"

class TestEnvironment : public testing::Environment {
public:
	virtual ~TestEnvironment() {}
	virtual void SetUp() {}

	void TearDown() {
		cudaDeviceReset(); //for cuda-memchecking
	}
};

// ROMEL 
// ral_resolution, reference_solution
void CHECK_RESULTS(std::vector<gdf_column_cpp> ral_solution, std::vector<std::string> reference_solution, std::vector<std::string> resultTypes)
{
    EXPECT_TRUE(ral_solution.size() == reference_solution.size() );

    /*for(int I=0; I<column.size(); I++)
    {
        auto column = referenceOutput.result[I];
        for(int J=0; J<column.size(); J++)
        {
            EXPECT_TRUE( outputs[I][J] == column[J]);
        }
    }*/
}

// ALEX
 std::vector<std::vector<gdf_column_cpp>> InputTablesFrom(
    std::vector<std::string> dataTypes, 
    std::vector<std::vector<std::string> > data) 
{
    std::vector<std::vector<gdf_column_cpp>> input_tables;
    return input_tables;    
}
class GeneratedTest : public testing::TestWithParam<Item> {};

TEST_P(GeneratedTest, RalOutputAsExpected) {

    std::cout<< GetParam().query << "|" << GetParam().logicalPlan<<std::endl;

    auto input_tables = InputTablesFrom(GetParam().data, GetParam().dataTypes);

    std::vector<gdf_column_cpp> outputs;
    // @todo: std::vector<std::string> tableNames{GetParam().tableName};
    std::vector<std::string> tableNames{};
    std::vector<std::vector<std::string>> columnNames = {};

    gdf_error err = evaluate_query(input_tables,
        tableNames,
        columnNames,
        GetParam().logicalPlan,
        outputs);
    EXPECT_TRUE(err == GDF_SUCCESS);
    CHECK_RESULTS(outputs, GetParam().result, GetParam().resultTypes);

}i

INSTANTIATE_TEST_CASE_P(
    TestsFromDisk,
    GeneratedTest,
    testing::ValuesIn( inputSet )
);

int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::Environment* const env = ::testing::AddGlobalTestEnvironment(new TestEnvironment());
    return RUN_ALL_TESTS();
}