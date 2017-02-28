# coding=utf-8

from pandas import Series, DataFrame
import pandas as pd
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas_datareader

np.set_printoptions(precision=4)

obj = Series([4, 7, -5, 3])
# print obj
# print obj.values
# print obj.index

obj = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
# print obj.index
# print obj[['a', 'c', 'a']]
# print obj[obj < 0]
# print obj*2

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
obj3

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
# print obj4
# print pd.notnull(obj4)
# print obj4.isnull()

# print obj4.index.name

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
# print frame

frame = DataFrame(data, columns=['year', 'state', 'pop'])
# print frame

frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])
# print frame2
# print frame2.columns
# print frame2['state']
# print frame2.year
# print frame2.ix['one']

pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}

frame3 = DataFrame(pop)
# print frame3.T

obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj = obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)
# print obj

obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'],
                  columns=['Ohio', 'Texas', 'California'])
# print frame

obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
# print new_obj
data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data = data.drop(['Colorado', 'Ohio'])
data = data.drop(['two', 'four'], axis=1)
# print data

obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
# print data.ix['Colorado', ['two', 'three']]

frame = DataFrame(np.random.randn(4, 3), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.ix[0]
# print series

frame = DataFrame(np.random.randn(4, 3), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', "Oregon"])

f_format = lambda x: '%.2f' % x
frame = frame.applymap(f_format)
# print frame

obj = Series(range(4), index=['d', 'a', 'b', 'c'])
obj = obj.sort_index()

frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'],
                  columns=['d', 'a', 'b', 'c'])
# print frame.sort_index(axis=1, ascending=False)

frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
# print frame.sort_index(by=['a', 'b'])

obj = Series([7, -5, 7, 4, 2, 0, 4])
# print obj.rank()

# axis DataFrame的行用1，列用0

df = DataFrame([[1.4, np.nan], [7.1, -4.5],
                [np.nan, np.nan], [0.75, -1.3]],
               index=['a', 'b', 'c', 'd'],
               columns=['one', 'two'])
# print df

obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
obj.unique()
# print obj.value_counts()

string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])

data = Series(np.random.randn(10),
              index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
                     [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
# print data[:, 3]
# print data.unstack()
frame = DataFrame(np.arange(16).reshape((4, 4)),
                  index=[['a', 'a', 'b', 'b']],
                  columns=[['Ohio', 'Ohio', 'Colorado', 'Colorado'],
                           ['Green', 'Red', 'Green', 'Red']])
print frame





