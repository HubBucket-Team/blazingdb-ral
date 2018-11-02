#!/bin/sh
CWD="$(pwd)"
echo "Build cpp library"
cd ../../resources/testdata-generator

python3 testdata.py sample.json ~/blazingdb/blazingdb-calcite/blazingdb-calcite-cli/target/BlazingCalciteCli.jar -O ../../tests/evaluate_query/evaluate_query.cu
cd $CWD
clang-format-3.8 -i -style=Google evaluate_query.cu
