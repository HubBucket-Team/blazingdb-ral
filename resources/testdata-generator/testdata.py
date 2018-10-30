#!/usr/bin/env python

import argparse
import json
import re
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser(description='Generate RAL test data.')
  parser.add_argument('filename', type=str,
                      help='Fixture JSON file name')
  parser.add_argument('calcite_jar', type=str,
                      help='Calcite CLI Jar')
  parser.add_argument('-O', '--output', type=str, default='-',
                      help='Output file path or - for stdout')
  args = parser.parse_args()

  items = make_items(args.filename)

  plans = make_plans(items, args.calcite_jar)

  strings_classes = (Φ(item, plan) for item, plan in zip(items, plans))

  header_text = '\n'.join(strings_classes)

  write(header_text).to(args.output)


def make_items(filename):
  with sys.stdin if '-' == filename else open(filename) as jsonfile:
    return [item_from(dct) for dct in json.load(jsonfile)]


def make_plans(items, calcite_jar):
  inputjson = lambda item: re.findall('non optimized\\n(.*)\\n\\noptimized',
      subprocess.Popen(('java', '-jar', calcite_jar),
                       stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE).communicate(json.dumps({
                         'columnNames': item.schema.columnNames,
                         'types': item.schema.columnTypes,
                         'name': item.schema.tableName,
                         'dbName': item.schema.dbName,
                         'query': item.query,
                       }).encode())[0].decode('utf-8'), re.M|re.S)[0]
  return (inputjson(item) for item in items)


def item_from(dct):
  return type('json_object', () ,{
    key: item_from(value) if type(value) is dict else value
    for key, value in dct.items()})


def Φ(item, plan):
  return ('InputTestItem{.query = "%(query)s", .logicalPlan ="%(plan)s",'
          ' .dataTable = %(dataTable)s, .resultTable = %(resultTable)s},') % {
  'query': item.query,
  'plan': '\\n'.join(line for line in plan.split('\n')),
  'dataTable': make_table(item.data, item.schema.tableName, item.schema.columnNames, item.schema.columnTypes),
  'resultTable': make_table(item.result, 'ResultSet', item.resultTypes, item.resultTypes),
  }


def make_table(data, tableName, columnNames, columnTypes):
  return ('LiteralTableBuilder{.name = "%(tableName)s",'
          ' .columns = %(literals)s}.Build()') % {
    'tableName': tableName,
    'literals': make_literals(data, columnNames, columnTypes),
  }


def make_literals(data, columnNames, columnTypes):
  return '{%s}' % (
    ','.join(['{.name = "%s", .values = Literals<%s>{%s} }'
              % (name, _type, ','.join(str(x) for x in values)[:-1])
              for name, _type, values in zip(columnNames, columnTypes, data)]))


def write(header_text):
  def to(filename):
    with sys.stdout if '-' == filename else open(filename, 'w') as output:
      output.write(HEADER_DEFINITIONS % header_text)
  return type('writer', (), dict(to=to))


HEADER_DEFINITIONS = '''
#pragma once

#include <string>
#include <vector>

#include <gdf/library/api.h>

using gdf::library::LiteralTableBuilder;
using gdf::library::Literal;

struct InputTestItem {
  std::string query;
  std::string logicalPlan;
  gdf::library::Table dataTable;
  gdf::library::Table resultTable;
};

std::vector<InputTestItem> inputTestSet{
%s
};

'''


if '__main__' == __name__:
  main()
