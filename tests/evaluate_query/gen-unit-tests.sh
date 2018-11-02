#!/bin/sh
CWD="$(pwd)"
echo "Build cpp library"
cd ../../resources/testdata-generator

#example: ~/blazingdb/blazingdb-calcite/blazingdb-calcite-cli/target/BlazingCalciteCli.jar
CalciteCli=$1
python3 testdata.py sample.json $CalciteCli -O ../../tests/evaluate_query/evaluate_query.cu
cd $CWD
clang-format-3.8 -i -style=Google evaluate_query.cu
