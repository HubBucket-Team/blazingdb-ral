#!/usr/bin/env python

import argparse
import json
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser(description='Generate RAL test data.')
  parser.add_argument('filename', type=str,
                      help='Fixture JSON file name')
  parser.add_argument('-O', '--output', type=str, default='-',
                      help='Output file path or - for stdout')
  args = parser.parse_args()

  items = make_items(args.filename)

  plans = make_plans(items)

  strings_classes = (Φ(item, plan) for item, plan in zip(items, plans))

  header_text = '\n'.join(strings_classes)

  write(header_text).to(args.output)


def make_items(filename):
  with sys.stdin if '-' == filename else open(filename) as jsonfile:
    return [item_from(dct) for dct in json.load(jsonfile)]


def make_plans(items):
  return (subprocess.check_output(('python',
                                   'calcite_fake.py',
                                   item.query,
                                   item.schema.dbName,
                                   item.schema.tableName,
                                   ','.join(item.schema.columnNames),
                                   ','.join(item.schema.columnTypes)))
          for item in items)


def item_from(dct):
  return type('json_object', () ,{
    key: item_from(value) if type(value) is dict else value
    for key, value in dct.items()})


def Φ(item, plan):
  return ('Item %(objectName)s'
          '{"%(query)s", "%(plan)s", %(dataTypes)s,'
          ' %(resultTypes)s, %(data)s, %(result)s};') % {
  'objectName': item.objectName,
  'query': item.query,
  'plan': '\\n'.join(line.decode() for line in plan.split(b'\n'))[2:-2],
  'dataTypes': '{%s}' % ','.join('"%s"' % str(columnType)
                                 for columnType in item.schema.columnTypes),
  'resultTypes': '{%s}' % ','.join('"%s"' % str(resultType)
                                   for resultType in item.resultTypes),
  'data': make_str(item.data),
  'result': make_str(item.result)[:-1]
  }


def make_str(collections):
  return ('{%s},' % ','.join('{%s}' % ','.join(('"%s"' % str(value)
                                                for value in collection))
                             for collection in collections))


def write(header_text):
  def to(filename):
    with sys.stdout if '-' == filename else open(filename, 'w') as output:
      output.write('\n'.join((HEADER_DEFINITIONS, header_text, '')))
  return type('writer', (), dict(to=to))


HEADER_DEFINITIONS = '''
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

'''


if '__main__' == __name__:
  main()
