#!/bin/sh
CWD="$(pwd)"
echo "Build cpp library"
cd resources/testdata-generator

python testdata.py sample.json > ../../tests/integration/generated.h
