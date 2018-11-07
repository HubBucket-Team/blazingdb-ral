from pydrill.client import PyDrill
from copy import deepcopy
from collections import OrderedDict
import tpch
import numpy as np

def get_bz_query(query, table_name):
  return query % {'table_name' : table_name}

def get_drill_query(query, table_name):
  return query % {'table_name' :  'dfs.tmp.`%(table)s`' % {'table': table_name} }

def get_reference_input(drill, test_name, query, table_name):
  ROOT_PATH = '/home/aocsa/blazingdb/tpch/1mb/'
  data_path = ROOT_PATH + table_name + '.psv'

  table = tpch.tables[table_name]
  drill_query = get_drill_query(query, table_name)
  return ('''
  {
      "testName": "%(test_name)s",
      "query": "%(query)s",
      "schema": {
        "dbName": "main",
        "tableName": "%(table_name)s",
        "columnNames": %(column_names)s,
        "columnTypes": %(column_types)s
      },
      "dataPath": "%(data_path)s",
      "selectedColumns": %(selected_columns)s,
      "result":  %(result)s
      "resultTypes": %(result_types)s
  }
  ''') % {'test_name': test_name,
          'query': get_bz_query(query, table_name),
          'table_name': table_name,
          'data_path':data_path,
          'column_names': get_column_names(table),
          'column_types': get_column_types(table),
          'selected_columns': get_selected_columns(table),
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
  return '[%s]' % (','.join([ get_gdf_type(table, name) for name in b3]))

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
  return '[%s]' % (','.join([_name for _name, _type in table.items()]))

def get_column_types(table):
  return '[%s]' % (','.join([ gdf_type( native_type(_type) ) for _name, _type in table.items()]))

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

  tpch.init_schema(drill)

  # query = 'SELECT c_custkey,  c_nationkey, c_acctbal FROM  %(table_name)s LIMIT 5'
  # res = get_reference_input(drill, 'TEST_01', query, 'customer')
  # print(res)

  query = 'select c_nationkey, count(c_custkey) from %(table_name)s group by c_nationkey, c_mktsegment'
  res = get_reference_input(drill, 'TEST_01', query, 'customer')
  print(res)
