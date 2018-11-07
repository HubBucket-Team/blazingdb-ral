from pydrill.client import PyDrill
from copy import deepcopy
from collections import OrderedDict
import tpch
import numpy as np
import re

def get_table_occurrences(query):
  # [ y for y in a if y not in b]
  return [name for name in tpch.tableNames if name in query.split()]

def replace_all(text, dic):
  for i, j in dic.items():
    text = re.sub(r"\s%s\s" % i, j, text)
  return text

def get_blazingsql_query(db_name, query):
  new_query = query
  for table_name in get_table_occurrences(query):
    new_query = replace_all(new_query, {table_name: ' %(table)s ' % {'table': db_name + '.' + table_name}})
  return new_query


def get_drill_query(query):
  new_query = query
  for table_name in get_table_occurrences(query):
    new_query = replace_all(new_query, {table_name: ' dfs.tmp.`%(table)s` ' % {'table': table_name}})
  return new_query


def get_reference_input(drill, root_path, test_name, query):
  table_names = get_table_occurrences(query)
  table_inputs = []
  for table_name in table_names:
    file_path = root_path + table_name + '.psv'
    table = tpch.tables[table_name]
    db_name = "main"
    table_inputs.append('''{
        "dbName": "%(db_name)s",
        "tableName": "%(table_name)s",
        "filePath": "%(file_path)s",
        "columnNames": %(column_names)s,
        "columnTypes": %(column_types)s
      }''' % {
          'db_name': db_name,
          'table_name': table_name,
          'file_path': file_path,
          'column_names': get_column_names(table),
          'column_types': get_column_types(table),
          })
  drill_query = get_drill_query(query)
  return ('''
  {
      "testName": "%(test_name)s",
      "query": "%(query)s",
      "tables": [%(table_inputs)s],
      "result":  %(result)s,
      "resultTypes": %(result_types)s
  }
  ''') % {'test_name': test_name,
          'query': get_blazingsql_query(db_name, query),
          'table_inputs': ','.join(table_inputs),
          'result': get_reference_result(drill, table, drill_query),
          'result_types': get_reference_result_types(drill, table, drill_query),
          }


def get_reference_result(drill, table, query_str):
  query_result = drill.query(query_str)
  df = query_result.to_dataframe()
  b1 = table.keys()
  b2 = list(df.columns.values)
  b3 = [val for val in b1 if val in b2]
  items = []
  for column in b3:
    s = '[%s]' % (','.join([item for item in np.asarray(df[column])]))
    items.append(s)

  return '[%s]' % (','.join(items))

def get_reference_result_types(drill, table, query_str):
  query_result = drill.query(query_str)
  data_frame = query_result.to_dataframe()
  b1 = table.keys()
  b2 = list(data_frame.columns.values)
  b3 = [val for val in b1 if val in b2]
  return '[%s]' % (','.join([ '"%s"' % get_gdf_type(table, name) for name in b3]))

def get_gdf_type(table, column_name):
  types = {
    'double': 'GDF_FLOAT64',
    'float': 'GDF_FLOAT32',
    'long': 'GDF_INT64',
    'int': 'GDF_INT32',
    'short': 'GDF_INT32',
    'char': 'GDF_INT8',
    'date': 'GDF_DATE64',
  }
  t = table.get(column_name)
  if t in  types:
    return types[t]
  return 'GDF_UNDEFINED'

def  get_column_names(table):
  return '[%s]' % (','.join([ '"%s"' % _name for _name, _type in table.items()]))

def get_column_types(table):
  return '[%s]' % (','.join([ '"%s"' % gdf_type( native_type(_type) ) for _name, _type in table.items()]))

def native_type(type_name):
  if type_name.find('string') != -1:
    return 'string'
  else:
    return type_name

def gdf_type(type_name):
   return {
    'double': 'GDF_FLOAT64',
    'float': 'GDF_FLOAT32',
    'long': 'GDF_INT64',
    'int': 'GDF_INT32',
    'short': 'GDF_INT32',
    'char': 'GDF_INT8',
    'date': 'GDF_INT64',
    'string': 'GDF_INT64',
  }[type_name]


def get_selected_columns(table):
  # [ y for y in a if y not in b]
  return  [_name for _name, _type in table.items() if _type.find('string') == -1 ]


if __name__ == '__main__':
  drill = PyDrill(host='localhost', port=8047)
  if not drill.is_active():
      raise Exception('Please run Drill first')

  ROOT_PATH = '/home/aocsa/blazingdb/tpch/1mb/'
  tpch.init_schema(drill, ROOT_PATH)
  query = '''select nation.n_nationkey, region.r_regionkey from nation inner join region on region.r_regionkey = nation.n_nationkey'''
  res = get_reference_input(drill, ROOT_PATH, 'TEST_01', query)
  print(res)
