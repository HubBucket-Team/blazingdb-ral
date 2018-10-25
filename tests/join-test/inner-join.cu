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
//#include <sqls_rtti_comp.hpp> //TODO build fails here it seems we need to export this header from libgdf

class TestEnvironment : public testing::Environment {
public:
	virtual ~TestEnvironment() {}
	virtual void SetUp() {}

	void TearDown() {
		cudaDeviceReset(); //for cuda-memchecking
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
 

template<class TypeParam>
void Check(gdf_column_cpp out_col, TypeParam* host_output){
	TypeParam * device_output;
	device_output = new TypeParam[out_col.size()];
	cudaMemcpy(device_output, out_col.data(), out_col.size() * sizeof(TypeParam), cudaMemcpyDeviceToHost);

	for(int i = 0; i < out_col.size(); i++){
		EXPECT_EQ(host_output[i], device_output[i]);
	}
}

template <class TypeParam>
struct InnerJoinTest : public ::testing::Test {
 
	void SetUp() {

		hr_emps.resize(3);
		hr_emps[0].create_gdf_column(DTypeTraits<TypeParam>::dtype, 3, (void *) emps_x, sizeof(TypeParam));
		hr_emps[1].create_gdf_column(DTypeTraits<TypeParam>::dtype, 3, (void *) emps_y, sizeof(TypeParam));
		hr_emps[2].create_gdf_column(DTypeTraits<TypeParam>::dtype, 3, (void *) emps_z, sizeof(TypeParam));

		hr_joiner.resize(2);
		hr_joiner[0].create_gdf_column(DTypeTraits<TypeParam>::dtype, 6, (void *) joiner_join_x, sizeof(TypeParam));
		hr_joiner[1].create_gdf_column(DTypeTraits<TypeParam>::dtype, 6, (void *) joiner_y, sizeof(TypeParam));
 
		input_tables.resize(2);
		input_tables[0] = hr_emps;
		input_tables[1] = hr_joiner;

	}
	void TearDown(){
		for(int i = 0; i < outputs.size(); i++){
			GDFRefCounter::getInstance()->free_if_deregistered(outputs[i].get_gdf_column());
		}
	}

	std::vector<std::vector<gdf_column_cpp> > input_tables;
	std::vector<gdf_column_cpp> hr_emps;
	std::vector<gdf_column_cpp> hr_joiner;

	TypeParam emps_x[3] = { 1, 2, 3};
	TypeParam emps_y[3] = { 4, 5, 6};
	TypeParam emps_z[3] = { 10, 10, 10};

	TypeParam joiner_join_x[6] = { 1, 1, 1, 2, 2, 3};
	TypeParam joiner_y[6] = { 1, 2, 3, 4 ,5 ,6};

	std::vector<std::string> table_names = { "hr.emps" , "hr.joiner"};
	std::vector<std::vector<std::string>> column_names = {{"x","y","z"},{"join_x","join_y"}};

	std::vector<gdf_column_cpp> outputs;
	std::vector<std::string> output_column_names;
	void * temp_space = nullptr;
};

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

TYPED_TEST_CASE(InnerJoinTest, DTypesTestTypes);

TYPED_TEST(InnerJoinTest, withGdfDType) {
 		// select *, x +joiner.y from hr.emps inner join hr.joiner on hr.joiner.join_x = hr.emps.x
		std::string query = "\
LogicalProject(x=[$0], y=[$1], z=[$2], join_x=[$3], y0=[$4], EXPR$5=[+($0, $4)])\n\
  LogicalJoin(condition=[=($3, $0)], joinType=[inner])\n\
    EnumerableTableScan(table=[[hr, emps]])\n\
    EnumerableTableScan(table=[[hr, joiner]])";

		gdf_error err = evaluate_query(this->input_tables, this->table_names, this->column_names,
				query, this->outputs);

		TypeParam out0[] = {1,1,1,2,2,3};
		TypeParam out1[] = {4,4,4,5,5,6};
		TypeParam out2[] = {10,10,10,10,10,10};
		TypeParam out3[] = {1,1,1,2,2,3};
		TypeParam out4[] = {1,2,3,4,5,6};
		TypeParam out5[] = {2,3,4,6,7,9};

		Check(this->outputs[0], out0);
		Check(this->outputs[1], out1);
		Check(this->outputs[2], out2);
		Check(this->outputs[3], out3);
		Check(this->outputs[4], out4);
		Check(this->outputs[5], out5);

		EXPECT_TRUE(err == GDF_SUCCESS);
} 



