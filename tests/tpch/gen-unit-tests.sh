#!/bin/sh
CWD="$(pwd)"
echo "Build cpp library"
cd ../../resources/tpch-generator

#example: ~/blazingdb/blazingdb-calcite/blazingdb-calcite-cli/target/BlazingCalciteCli.jar
CalciteCli=$1
python3  test_generator.py tpch_queries.json $CalciteCli -O ../../tests/tpch/tpch-tests.cu
cd $CWD
clang-format -i -style=Google tpch-tests.cu
