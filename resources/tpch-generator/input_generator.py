from pydrill.client import PyDrill
from copy import deepcopy
from collections import OrderedDict
import tpch
import numpy as np
import pandas as pd
import re
import argparse
import sys

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

  print('\t#drill_query:   ', drill_query)
  return ('''
  {
      "testName": "%(test_name)s",
      "query": "%(query)s",
      "tables": [%(table_inputs)s],
      "result":  %(result)s,
      "resultTypes": %(result_types)s,
      "resultColumnNames": %(result_names)s
  }
  ''') % {'test_name': test_name,
          'query': get_blazingsql_query(db_name, query),
          'table_inputs': ','.join(table_inputs),
          'result': get_reference_result(drill, table, drill_query),
          'result_types': get_reference_result_types(drill, table, drill_query),
          'result_names': get_reference_result_names(drill, table, drill_query),
          }

def get_reference_result_names(drill, table, query_str):
  query_result = drill.query(query_str)
  return  '[%s]' % (','.join(['"%s"' % name for name in query_result.columns]))

def get_reference_result(drill, table, query_str):
  query_result = drill.query(query_str)
  df = query_result.to_dataframe()
  items = []
  for column in query_result.columns:
    s = '[%s]' % (','.join([item for item in np.asarray(df[column])]))
    items.append(s)

  return '[%s]' % (','.join(items))

def get_reference_result_types(drill, table, query_str):
  query_result = drill.query(query_str)
  df = query_result.to_dataframe()
  items = []
  def get_dtype(dtype):
    if pd.api.types.is_categorical_dtype(dtype):
      return 'GDF_INT8'
    dicc = {np.float64: 'GDF_FLOAT64',
               np.float32: 'GDF_FLOAT32',
               np.int64:   'GDF_INT64',
               np.int32:   'GDF_INT32',
               np.int16:   'GDF_INT16',
               np.int8:    'GDF_INT8',
               np.bool_:   'GDF_INT8',
               np.datetime64: 'GDF_DATE64',
               np.object: 'GDF_INT64'
               }
    _type = np.dtype(dtype).type
    if _type in dicc :
      return dicc[_type]
    return 'GDF_INT64'
  return '[%s]' % (','.join([ '"%s"' % get_dtype(df[column].dtype) for column in query_result.columns]))

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



def write(json_list):
    def to(filename):
        with sys.stdout if '-' == filename else open(filename, 'w') as output:
            output.write( '[%s]' % (','.join([ item for item in json_list])) )
    return type('writer', (), dict(to=to))

if __name__ == '__main__':
  drill = PyDrill(host='localhost', port=8047)
  if not drill.is_active():
      raise Exception('Please run Drill first')

  parser = argparse.ArgumentParser(description='Generate Input Generator for UnitTestGenerator.')
  parser.add_argument('tpch_path', type=str,
                      help='use complete path, ex /tmp/tpch/1mb/')
  parser.add_argument('-O', '--output', type=str, default='-',
                      help='Output file path or - for stdout')
  args = parser.parse_args()

  tpch_path = args.tpch_path
  tpch.init_schema(drill, tpch_path)

  aggregration_queries = [
    'select count(p_partkey), sum(p_partkey), avg(p_partkey), max(p_partkey), min(p_partkey) from part'
  ]

  groupby_queries = [
    'select count(c_custkey), sum(c_acctbal), avg(c_acctbal), min(c_custkey), max(c_nationkey), c_nationkey from customer group by c_nationkey',
    'select count(c_custkey), sum(c_acctbal), avg(c_acctbal), min(c_custkey), max(c_custkey), c_nationkey from customer where c_custkey < 50 group by c_nationkey',
    # 'select count(c_custkey) + sum(c_acctbal) + avg(c_acctbal), min(c_custkey) - max(c_nationkey), c_nationkey * 2 as key from customer where key < 40 group by key'
  ]
  orderby_queries = [
    'select c_custkey, c_acctbal from customer order by c_acctbal',
    'select c_custkey, c_nationkey, c_acctbal from customer order by c_acctbal',
    'select c_custkey, c_nationkey, c_acctbal from customer order by c_nationkey, c_acctbal',
    'select c_custkey, c_nationkey, c_acctbal from customer order by c_nationkey, c_custkey'
  ]
  join_queries = [
    'select nation.n_nationkey, region.r_regionkey from nation inner join region on region.r_regionkey = nation.n_nationkey',
    'select avg(c.c_custkey), avg(c.c_nationkey), n.n_regionkey from customer as c inner join nation as n on c.c_nationkey = n.n_nationkey group by n.n_regionkey',
    'select c.c_custkey, c.c_nationkey, n.n_regionkey from customer as c inner join nation as n on c.c_nationkey = n.n_nationkey where n.n_regionkey = 1 and c.c_custkey < 50',
    'select avg(c.c_custkey), avg(c.c_acctbal), n.n_nationkey, r.r_regionkey from customer as c inner join nation as n on c.c_nationkey = n.n_nationkey inner join region as r on r.r_regionkey = n.n_regionkey group by n.n_nationkey, r.r_regionkey',
    'select n1.n_nationkey as supp_nation, n2.n_nationkey as cust_nation, l.l_extendedprice * l.l_discount from supplier as s inner join lineitem as l on s.s_suppkey = l.l_suppkey inner join orders as o on o.o_orderkey = l.l_orderkey inner join customer as c on c.c_custkey = o.o_custkey inner join nation as n1 on s.s_nationkey = n1.n_nationkey inner join nation as n2 on c.c_nationkey = n2.n_nationkey where n1.n_nationkey = 1 and n2.n_nationkey = 2 and o.o_orderkey < 10000',
  ]
  where_queries = [
    'select c_custkey, c_nationkey, c_acctbal from customer where c_custkey < 15',
    'select c_custkey, c_nationkey, c_acctbal from customer where c_custkey < 150 and c_nationkey = 5',
    'select c_custkey, c_nationkey as nkey from customer where c_custkey < 0',
    'select c_custkey, c_nationkey as nkey from customer where c_custkey < 0 and c_nationkey >=30',
    'select c_custkey, c_nationkey as nkey from customer where c_custkey < 0 or c_nationkey >= 24',
    'select c_custkey, c_nationkey as nkey from customer where c_custkey < 0 and c_nationkey >= 3',
    'select c_custkey, c_nationkey as nkey from customer where -c_nationkey + c_acctbal > 750.3',
    'select c_custkey, c_nationkey as nkey from customer where -c_nationkey + c_acctbal > 750'
    ]

  issues_query = [
    'select count(c_custkey), c_nationkey, count(c_acctbal) from customer group by c_nationkey'
    ]

  json_list = []
  for index, query in enumerate(where_queries):
    print('## processing...\t', query)
    json_text = get_reference_input(drill, tpch_path, 'TEST_0%s' % index , query)
    json_list.append(json_text)
  write(json_list).to('where_queries.json')

  json_list = []
  for index, query in enumerate(join_queries):
    print('## processing...\t', query)
    json_text = get_reference_input(drill, tpch_path, 'TEST_0%s' % index , query)
    json_list.append(json_text)
  write(json_list).to('join_queries.json')

  json_list = []
  for index, query in enumerate(orderby_queries):
    print('## processing...\t', query)
    json_text = get_reference_input(drill, tpch_path, 'TEST_0%s' % index , query)
    json_list.append(json_text)
  write(json_list).to('orderby_queries.json')

  json_list = []
  for index, query in enumerate(orderby_queries):
    print('## processing...\t', query)
    json_text = get_reference_input(drill, tpch_path, 'TEST_0%s' % index , query)
    json_list.append(json_text)
  write(json_list).to('groupby_queries.json')
