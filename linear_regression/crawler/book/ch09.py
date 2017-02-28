# coding=utf-8

from __future__ import division
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas as pd
np.set_printoptions(precision=4)

df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                'key2': ['one', 'two', 'one', 'two', 'one'],
                'data1': np.random.randn(5),
                'data2': np.random.randn(5)})
grouped = df['data1'].groupby(df['key1'])
# print grouped.count()

# print df['data1'].groupby([df['key1'], df['key2']]).mean().unstack()

states = np.array(['Ohio', 'Ohio', 'California', 'California', 'Ohio'])
years = np.array([2005, 2006, 2005, 2006, 2005])
# print df['data1'].groupby([states, years]).mean().unstack()

# for name, group in df.groupby('key1'):
#     print name
#     print group

# for (k1, k2), group in df.groupby(['key1', 'key2']):
#     print (k1, k2)
#     print group

pieces = dict(list(df.groupby('key1')))
# print pieces

# print df.groupby(['key1', 'key2'])['data2'].mean().unstack()

people = DataFrame(np.random.randn(5, 5),
                   columns=['a', 'b', 'c', 'd', 'e'],
                   index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people.ix[2:3, ['b', 'c']] = np.nan  # Add a few NA values
mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
           'd': 'blue', 'e': 'red', 'f': 'orange'}
by_column = people.groupby(mapping, axis=1)
# print by_column.sum()

map_series = Series(mapping)
# print people.groupby(map_series, axis=1).count()

columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                    [1, 3, 5, 1, 3]], names=['cty', 'tenor'])
hier_df = DataFrame(np.random.randn(4, 5), columns=columns)
# print hier_df.groupby(level='cty', axis=1).count()


def peak_to_peak(arr):
    return arr.max() - arr.min()

grouped = df.groupby('key1')
# print grouped.agg(peak_to_peak)

tips = pd.read_csv('ch06/tips.csv')
tips['tip_pct'] = tips['tip'] / tips['total_bill']
# print tips[:6]
grouped = tips.groupby(['sex', 'smoker'])
grouped_pct = grouped['tip_pct']
# print grouped_pct.agg(['mean', 'std', peak_to_peak])
# print grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])

functions = ['count', 'mean', 'max']
result = grouped['tip_pct', 'total_bill'].agg(functions)
# print result

# print tips.groupby(['sex', 'smoker'], as_index=False).mean()

k1_means = df.groupby('key1').mean().add_prefix('mean_')
# print pd.merge(df, k1_means, left_on='key1', right_on='key1', right_index=True)

key = ['one', 'two', 'one', 'two', 'one']
# print people.groupby(key).mean()
# print people.groupby(key).transform(np.mean)


def demean(arr):
    return arr - arr.mean()

demeaned = people.groupby(key).transform(demean)
# print demeaned
# print demeaned.groupby(key).mean()


def top(df, n=5, column='tip_pct'):
    return df.sort_values(by=column)[-n:]

# print top(tips, n=6)
# print tips.groupby('smoker').apply(top)
# print tips.groupby(['smoker', 'day']).apply(top, n=2, column='total_bill')

frame = DataFrame({'data1': np.random.randn(1000),
                   'data2': np.random.randn(1000)})
factor = pd.cut(frame.data1, 4)


def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}

grouped = frame.data2.groupby(factor)
grouped.apply(get_stats).unstack()

grouping = pd.qcut(frame.data1, 10, labels=False)
grouped = frame.data2.groupby(grouping)
# print grouped.apply(get_stats).unstack()

s = Series(np.random.randn(6))
s[::2] = np.nan
s.fillna(s.mean())

states = ['Ohio', 'New York', 'Vermont', 'Florida',
          'Oregon', 'Nevada', 'California', 'Idaho']
group_key = ['East'] * 4 + ['West'] * 4
data = Series(np.random.randn(8), index=states)
data[['Vermont', 'Nevada', 'Idaho']] = np.nan
# print data.groupby(group_key).mean()

suits = ['H', 'S', 'C', 'D']
card_val = (range(1, 11) + [10] * 3) * 4
base_names = ['A'] + range(2, 11) + ['J', 'K', 'Q']
cards = []
for suit in ['H', 'S', 'C', 'D']:
    cards.extend(str(num) + suit for num in base_names)

deck = Series(card_val, index=cards)


def draw(deck, n=5):
    return deck.take(np.random.permutation(len(deck))[:n])
get_suit = lambda card: card[-1]
# print deck.groupby(get_suit).apply(draw, n=2)

# df = DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
#                 'data': np.random.randn(8),
#                 'weights': np.random.rand(8)})
# grouped = df.groupby('category')
# get_wavg = lambda g: np.average(g['data'], weights=['weights'])

close_px = pd.read_csv('ch06/stock_px.csv', parse_dates=True, index_col=0)
# print close_px

tips.pivot_table(['tip_pct', 'size'], index=['sex', 'day'])
tips.pivot_table('size', index=['time', 'sex', 'smoker'],
                 columns=['day'], aggfunc='sum', fill_value=0)

from StringIO import StringIO
data = """\
Sample    Gender    Handedness
1    Female    Right-handed
2    Male    Left-handed
3    Female    Right-handed
4    Male    Right-handed
5    Male    Left-handed
6    Male    Right-handed
7    Female    Right-handed
8    Female    Left-handed
9    Male    Right-handed
10    Female    Right-handed"""
data = pd.read_table(StringIO(data), sep='\s+')

pd.crosstab(data.Gender, data.Handedness)
# print pd.crosstab([tips.time, tips.day], tips.smoker, margins=True)

fec = pd.read_csv('ch06/P00000001-ALL.csv')
# print fec.info()
# print fec.ix[123]
unique_cands = fec.cand_nm.unique()
# print unique_cands
parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}
# print fec.cand_nm[1:10]
# print fec.cand_nm[10:100].map(parties)
fec['party'] = fec.cand_nm.map(parties)
# print fec['party'].value_counts()
fec = fec[fec.contb_receipt_amt > 0]
fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]
# print fec.contbr_occupation.value_counts()[:10]
occ_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS': 'NOT PROVIDED',
   'INFORMATION REQUESTED': 'NOT PROVIDED',
   'INFORMATION REQUESTED (BEST EFFORTS)': 'NOT PROVIDED',
   'C.E.O.': 'CEO'
}

# If no mapping provided, return x
f = lambda x: occ_mapping.get(x, x)
fec.contbr_occupation = fec.contbr_occupation.map(f)

emp_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS': 'NOT PROVIDED',
   'INFORMATION REQUESTED': 'NOT PROVIDED',
   'SELF' : 'SELF-EMPLOYED',
   'SELF EMPLOYED': 'SELF-EMPLOYED',
}

# If no mapping provided, return x
f = lambda x: emp_mapping.get(x, x)
fec.contbr_employer = fec.contbr_employer.map(f)






