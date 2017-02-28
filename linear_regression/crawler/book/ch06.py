# coding=utf-8

from __future__ import division
from numpy.random import randn
import numpy as np
import os
import sys
import csv
import json
import sqlite3
import requests
import matplotlib.pyplot as plt
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas as pd
from lxml.html import parse
from lxml import objectify
from urllib2 import urlopen
np.set_printoptions(precision=4)

df = pd.read_csv('ch06/ex1.csv')
df = pd.read_table('ch06/ex1.csv', sep=',')
# header 用作列明的行号，默认第一行
df = pd.read_csv('ch06/ex2.csv', header=None)
df = pd.read_csv('ch06/ex2.csv', names=['a', 'b', 'c', 'd', 'message'], index_col='message')
# print df

parsed = pd.read_csv('ch06/csv_mindex.csv', index_col=['key1', 'key2'])
# print parsed

# 有数量不定的空格分割的文件
result = pd.read_table('ch06/ex3.txt', sep='\s+')
# print result

# 利用skiprows跳过文件的某几行
# print pd.read_csv('ch06/ex4.csv', skiprows=[0, 2, 3])

# 缺失值处理
result = pd.read_csv('ch06/ex5.csv', na_values=['NULL'])
# print result

# 缺失值map
sentinels = {'message': ['foo', 'NA'], 'something': 'two'}
# print pd.read_csv('ch06/ex5.csv', na_values=sentinels)

# result = pd.read_csv('ch06/ex6.csv', nrows=5)
result = pd.read_csv('ch06/ex6.csv', chunksize=1000)
tot = Series([])
for piece in result:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)
# print tot.order(ascending=False)[:10]

data = pd.read_csv('ch06/ex5.csv')
# data.to_csv(sys.stdout, sep='|', na_rep='NULL')

dates = pd.date_range('1/1/2000', periods=10)
ts = Series(np.arange(10), index=dates)
# ts.to_csv('ch06/tseries.csv')
# print Series.from_csv('ch06/tseries.csv', parse_dates=True)

f = open('ch06/ex7.csv')
reader = csv.reader(f)
# for line in reader:
#     print line

lines = list(csv.reader(open('ch06/ex7.csv')))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}
# print data_dict

obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
              {"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""
result = json.loads(obj)
siblings = DataFrame(result['siblings'], columns=['name', 'age'])
# print siblings

parsed = parse(urlopen('http://www.baidu.com'))
doc = parsed.getroot()
links = doc.findall('.//a')
# for lnk in links:
#     print lnk.get('href')
#     print lnk.text_content()

query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
 c REAL,        d INTEGER
);"""

con = sqlite3.connect(':memory:')
con.execute(query)
con.commit()

data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"

con.executemany(stmt, data)
con.commit()

cursor = con.execute('select * from test')
rows = cursor.fetchall()
print rows



