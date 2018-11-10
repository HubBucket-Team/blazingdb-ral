from pydrill.client import PyDrill
from copy import deepcopy
from collections import OrderedDict
import tpch
import numpy as np
import pandas as pd
import re
import argparse
import sys
import input_generator as generator

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
        # 'select count(p_partkey), sum(p_partkey), avg(p_partkey), max(p_partkey), min(p_partkey) from part'
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

    # issues_query = [
    #     'select count(c_custkey), c_nationkey, count(c_acctbal) from customer group by c_nationkey'
    # ]

    generator.generate_json_input(drill, tpch_path,  where_queries, 'json_inputs/where_queries.json')
    generator.generate_json_input(drill, tpch_path, join_queries, 'json_inputs/join_queries.json')
    generator.generate_json_input(drill, tpch_path, orderby_queries, 'json_inputs/orderby_queries.json')
    generator.generate_json_input(drill, tpch_path, groupby_queries, 'json_inputs/groupby_queries.json')
    generator.generate_json_input(drill, tpch_path, aggregration_queries, 'json_inputs/aggregration_queries.json')
    
    all_queries = where_queries + join_queries + orderby_queries + groupby_queries + aggregration_queries
    generator.generate_json_input(drill, tpch_path, all_queries, 'json_inputs/all_queries.json')
    
