# coding=utf-8

from __future__ import division
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas
import pandas as pd
import re
import json
np.set_printoptions(precision=4, threshold=500)
pd.options.display.max_rows = 100

df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df2 = DataFrame({'key': ['a', 'b', 'd'],
                 'data2': range(3)})
# print pd.merge(df1, df2, how='outer')

df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                 'data1': range(6)})
df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
                 'data2': range(5)})
# print pd.merge(df1, df2, on='key', how='left')

s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])
s4 = pd.concat([s1 * 5, s3])

# print pd.concat([s1, s1, s3])

a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b = Series(np.arange(len(a), dtype=np.float64),
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan

# print range(2, 18, 4)

data = DataFrame(np.arange(6).reshape((2, 3)),
                 index=pd.Index(['Ohio', 'Colorado'], name='state'),
                 columns=pd.Index(['one', 'two', 'three'], name='number'))
# print data
result = data.stack()
# print result
result = result.unstack('state')
# print result

data = pd.read_csv('ch06/macrodata.csv')
periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date')
data = DataFrame(data.to_records(),
                 columns=pd.Index(['realgdp', 'infl', 'unemp'], name='item'),
                 index=periods.to_timestamp('D', 'end'))

ldata = data.stack().reset_index().rename(columns={0: 'value'})
wdata = ldata.pivot('date', 'item', 'value')
# print ldata[:10]
# print wdata[:10]

data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                  'k2': [1, 1, 2, 3, 3, 4, 4]})
data['v1'] = range(7)
# print data.drop_duplicates(['k1', 'k2'], keep='last')

data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami',
                           'corned beef', 'Bacon', 'pastrami', 'honey ham',
                           'nova lox'],
                  'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)

data = Series([1., -999., 2., -999., -1000., 3.])

data = DataFrame(np.arange(12).reshape((3, 4)),
                 index=['Ohio', 'Colorado', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data.index = data.index.map(str.upper)
# print data.rename(index=str.title, columns=str.upper)

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins, right=False)
# print pd.value_counts(cats)

data = np.random.rand(20)
data = pd.cut(data, 4, precision=2)

data = np.random.randn(1000)
cats = pd.qcut(data, 4)
# print pd.value_counts(cats)

np.random.seed(12345)
data = DataFrame(np.random.randn(1000, 4))

df = DataFrame(np.arange(5 * 4).reshape((5, 4)))
sampler = np.random.permutation(5)
# print sampler

val = 'a,b,  guido'
val.split(',')
pieces = [x.strip() for x in val.split(',')]
# print pieces

# 正则包括三部分：模式匹配、替换和拆分
# match findall search
text = "foo    bar\t baz  \tqux"
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
regex = re.compile(pattern, flags=re.IGNORECASE)


db = json.load(open('ch06/foods-2011-10-03.json'))
info_keys = ['description', 'group', 'id', 'manufacturer']
# 食物的名称，分类、编号、制造商信息
info = DataFrame(db, columns=info_keys)
# 食物类别的分布
# print pd.value_counts(info.group)[:10]

# 食物的营养成分整合
nutrients = []
for rec in db:
    fnuts = DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)
# print nutrients

nutrients = nutrients.drop_duplicates()
col_mapping = {'description': 'food', 'group': 'fgroup'}
info = info.rename(columns=col_mapping, copy=False)
col_mapping = {'description': 'nutrient', 'group': 'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
# print nutrients

ndata = pd.merge(nutrients, info, on='id', how='outer')
print ndata

# result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)
# result['Zinc, Zn'].order().plot(kind='barh')
# plt.show()




