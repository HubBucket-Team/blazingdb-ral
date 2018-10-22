// #include <cstdlib>
// #include <iostream>
// #include <vector>
// #include <string>

// #include "gtest/gtest.h"
// #include <CalciteExpressionParsing.h>
// #include <CalciteInterpreter.h>
// #include <StringUtil.h>
// #include <DataFrame.h>
// #include <GDFColumn.cuh>
// #include <GDFCounter.cuh>
// #include <Utils.cuh>

// #include <gdf/gdf.h>
// //ToDo Script for generate and/or update this header while cmaking
// #include "generated.h"

// class TestEnvironment : public testing::Environment {
// public:
// 	virtual ~TestEnvironment() {}
// 	virtual void SetUp() {}

// 	void TearDown() {
// 		cudaDeviceReset(); //for cuda-memchecking
// 	}
// };

// // ROMEL 
// // ral_resolution, reference_solution
// void CHECK_RESULTS(const std::vector<gdf_column_cpp>& ral_solution, const std::vector<std::string>& reference_solution, const std::vector<std::string> & resultTypes)
// {
//     EXPECT_TRUE(ral_solution.size() == reference_solution.size() );

//     /*for(int I=0; I<column.size(); I++)
//     {
//         auto column = referenceOutput.result[I];
//         for(int J=0; J<column.size(); J++)
//         {
//             EXPECT_TRUE( outputs[I][J] == column[J]);
//         }
//     }*/
// }


// namespace helper {
//     template <gdf_dtype>
//     struct GdfEnumType;

//     template <>
//     struct GdfEnumType<GDF_invalid> {
//         using Type = void;
//     };

//     template <>
//     struct GdfEnumType<GDF_INT8> {
//         using Type = int8_t;
//     };

//     template <>
//     struct GdfEnumType<GDF_INT16> {
//         using Type = int16_t;
//     };

//     template <>
//     struct GdfEnumType<GDF_INT32> {
//         using Type = int32_t;
//     };

//     template <>
//     struct GdfEnumType<GDF_INT64> {
//         using Type = int64_t;
//     };

//     template <>
//     struct GdfEnumType<GDF_UINT8> {
//         using Type = uint8_t;
//     };

//     template <>
//     struct GdfEnumType<GDF_UINT16> {
//         using Type = uint16_t;
//     };

//     template <>
//     struct GdfEnumType<GDF_UINT32> {
//         using Type = uint32_t;
//     };

//     template <>
//     struct GdfEnumType<GDF_UINT64> {
//         using Type = uint64_t;
//     };

//     template <>
//     struct GdfEnumType<GDF_FLOAT32> {
//         using Type = float;
//     };

//     template <>
//     struct GdfEnumType<GDF_FLOAT64> {
//         using Type = double;
//     };
// }

// namespace helper {
//     template <typename T>
//     struct GdfDataType;

//     template <>
//     struct GdfDataType<int8_t> {
//         static constexpr gdf_dtype Value = GDF_INT8;
//     };

//     template <>
//     struct GdfDataType<int16_t> {
//         static constexpr gdf_dtype Value = GDF_INT16;
//     };

//     template <>
//     struct GdfDataType<int32_t> {
//         static constexpr gdf_dtype Value = GDF_INT32;
//     };

//     template <>
//     struct GdfDataType<int64_t> {
//         static constexpr gdf_dtype Value = GDF_INT64;
//     };

//     template <>
//     struct GdfDataType<uint8_t> {
//         static constexpr gdf_dtype Value = GDF_UINT8;
//     };

//     template <>
//     struct GdfDataType<uint16_t> {
//         static constexpr gdf_dtype Value = GDF_UINT16;
//     };

//     template <>
//     struct GdfDataType<uint32_t> {
//         static constexpr gdf_dtype Value = GDF_UINT32;
//     };

//     template <>
//     struct GdfDataType<uint64_t> {
//         static constexpr gdf_dtype Value = GDF_UINT64;
//     };

//     template <>
//     struct GdfDataType<float> {
//         static constexpr gdf_dtype Value = GDF_FLOAT32;
//     };

//     template <>
//     struct GdfDataType<double> {
//         static constexpr gdf_dtype Value = GDF_FLOAT64;
//     };
// }

// template <gdf_dtype gdf_type>
// using GdfEnumType = typename helper::GdfEnumType<gdf_type>::Type;

// template <typename Type>
// using GdfDataType = helper::GdfDataType<Type>;


// gdf_dtype getType(std::string type) {
//     std::map<std::string, gdf_dtype> types {
//             {"GDF_INT8": GDF_INT8  },
//             {"GDF_INT16": GDF_INT16  },
//             {"GDF_INT32": GDF_INT32  },
//             {"GDF_INT64": GDF_INT64  },
//             {"GDF_UINT8": GDF_UINT8  },
//             {"GDF_UINT16": GDF_UINT16  },
//             {"GDF_UINT32": GDF_UINT32  },
//             {"GDF_UINT64": GDF_UINT64  },
//             {"GDF_DATE32": GDF_DATE32  },
//             {"GDF_DATE64": GDF_DATE64  },
//             {"GDF_TIMESTAMP": GDF_TIMESTAMP}                
//     };
//     return types[type];
// }


// gdf_column_cpp get_gdf_column_cpp (std::string type, std::vector<std::string> values) {

//   char * input1;
//   size_t num_values = 32;
//   input1 = new char[num_values]; // @todo, error
//   for(int i = 0; i < num_values; i++){
//     input1[i] = i;
//   }
//   one.create_gdf_column(gdf::GDF_INT8, num_values, (void *) input1, sizeof(int8_t));
// }

// // ALEX
//  std::vector<std::vector<gdf_column_cpp>> InputTablesFrom(
//     const std::vector<std::string> & dataTypes, 
//     const std::vector<std::vector<std::string>> & data) 
// {
//     std::vector<std::vector<gdf_column_cpp>> input_tables;

//     // std::vector<gdf_column_cpp> table;
//     // for(size_t i = 0; i < dataTypes.size(); i++)
//     // {
//     //     auto type = dataTypes[i];
//     //     auto column = data[i];
//     //     auto gdf_column_obj = get_gdf_column_cpp( GdfEnumType< getType > type, column);
//     //     table.push_back(gdf_column_obj);
//     // }
//     // input_tables.push_back(table);
//     return input_tables;    
// }
// class GeneratedTest : public testing::TestWithParam<Item> {};

// TEST_P(GeneratedTest, RalOutputAsExpected) {

//     std::cout<< GetParam().query << "|" << GetParam().logicalPlan<<std::endl;

//     auto input_tables = InputTablesFrom(GetParam().data, GetParam().dataTypes);

//     std::vector<gdf_column_cpp> outputs;
//     // @todo: std::vector<std::string> tableNames{GetParam().tableName};
//     std::vector<std::string> tableNames{};
//     std::vector<std::vector<std::string>> columnNames = {};

//     gdf_error err = evaluate_query(input_tables,
//         tableNames,
//         columnNames,
//         GetParam().logicalPlan,
//         outputs);
//     EXPECT_TRUE(err == GDF_SUCCESS);
//     CHECK_RESULTS(outputs, GetParam().result, GetParam().resultTypes);

// }i

// INSTANTIATE_TEST_CASE_P(
//     TestsFromDisk,
//     GeneratedTest,
//     testing::ValuesIn( inputSet )
// );

// int main(int argc, char **argv){
//     ::testing::InitGoogleTest(&argc, argv);
//     ::testing::Environment* const env = ::testing::AddGlobalTestEnvironment(new TestEnvironment());
//     return RUN_ALL_TESTS();
// }