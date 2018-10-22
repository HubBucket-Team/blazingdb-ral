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

struct RalItem {
    std::string logicalPlan;
    std::vector<std::vector<gdf_column_cpp> > input_tables;
    std::vector<std::string> table_names;
    std::vector<std::vector<std::string>> column_names;
    std::vector<gdf_column_cpp> outputs;
};

RalItem fromItem(Item item)
{
    RalItem ralItem;
    //ralItem.input_tables = 
    ralItem.table_names = std::vector<std::string> ( item.dataTypes.size() );
    ralItem.column_names = { std::vector<std::string> ( item.dataTypes.size() ) };
    ralItem.logicalPlan = item.logicalPlan;
    return ralItem;
}

void CHECK_RESULTS(std::vector<gdf_column_cpp> outputs, Item referenceOutput)
{
    EXPECT_TRUE( outputs.size() == referenceOutput.result.size() );

    /*for(int I=0; I<column.size(); I++)
    {
        auto column = referenceOutput.result[I];
        for(int J=0; J<column.size(); J++)
        {
            EXPECT_TRUE( outputs[I][J] == column[J]);
        }
    }*/
}

class GeneratedTest : public testing::TestWithParam<Item> {};

TEST_P(GeneratedTest, RalOutputAsExpected) {
    {
        std::cout<<GetParam().logicalPlan<<std::endl;

        EXPECT_TRUE(true);

        //auto ralItem = fromItem();

        //gdf_error err = evaluate_query(ralItem.input_tables, ralItem.table_names, ralItem.column_names, ralItem.logicalPlan, ralItem.outputs);
        //EXPECT_TRUE(err == GDF_SUCCESS);

        //CHECK_RESULTS(outputs, ralItem);
    }
}

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