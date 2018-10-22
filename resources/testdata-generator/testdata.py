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
  parser.add_argument('h2_path', type=str,
                      help='Path with H2 database folder')
  parser.add_argument('-O', '--output', type=str, default='-',
                      help='Output file path or - for stdout')
  args = parser.parse_args()

  items = make_items(args.filename)

  plans = make_plans(items, args.calcite_jar, args.h2_path)

  strings_classes = (Φ(item, plan) for item, plan in zip(items, plans))

  header_text = '\n'.join(strings_classes)

  write(header_text).to(args.output)


def make_items(filename):
  with sys.stdin if '-' == filename else open(filename) as jsonfile:
    return [item_from(dct) for dct in json.load(jsonfile)]


def make_plans(items, calcite_jar, h2_path):
  inputjson = lambda item: re.findall('non optimized\\n(.*)\\n\\noptimized',
      subprocess.Popen(('java', '-jar', calcite_jar),
                       cwd=h2_path,
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
  return ('Item{"%(query)s", "%(plan)s", %(dataTypes)s,'
          ' %(resultTypes)s, %(data)s, %(result)s},') % {
  'query': item.query,
  'plan': '\\n'.join(line for line in plan.split('\n')),
  'dataTypes': '{%s}' % ','.join('"%s"' % str(columnType)
                                 for columnType in item.schema.columnTypes),
  'resultTypes': '{%s}' % ','.join('"%s"' % str(resultType)
                                   for resultType in item.resultTypes),
  'data': make_str(item.data),
  'result': make_str(item.result)
  }


def make_str(collections):
  return ('{%s},' % ','.join('{%s}' % ','.join(('"%s"' % str(value)
                                                for value in collection))
                             for collection in collections))[:-1]


def write(header_text):
  def to(filename):
    with sys.stdout if '-' == filename else open(filename, 'w') as output:
      output.write(HEADER_DEFINITIONS % header_text)
  return type('writer', (), dict(to=to))


HEADER_DEFINITIONS = '''
#pragma once

#include <string>
#include <vector>

struct Item {
  std::string query;
  std::string logicalPlan;
  std::vector<std::string> dataTypes;
  std::vector<std::string> resultTypes;
  std::vector<std::vector<std::string> > data;
  std::vector<std::vector<std::string> > result;
};

std::vector<Item> inputSet{
%s
};

'''


if '__main__' == __name__:
  main()
