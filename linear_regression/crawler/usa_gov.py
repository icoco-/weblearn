# coding=utf-8


from __future__ import division
from collections import defaultdict
from collections import Counter
from pandas import Series, DataFrame
import pandas as pd
import json
from numpy.random import randn
import numpy as np
pd.options.display.max_rows = 12
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt


path = 'movie_data/usagov_bitly_data.txt'
# print open(path).readline()
records = [json.loads(line) for line in open(path)]
time_zones = [rec['tz'] for rec in records if 'tz' in rec]


def get_counts(seq):
    counts = defaultdict(int)
    for x in seq:
        counts[x] += 1
    return counts

counts = get_counts(time_zones)


def top_counts(seq, n=10):
    value_key_pairs = [(count, tz) for tz, count in seq.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

# print top_counts(counts, 10)

counts = Counter(time_zones)
# print counts.most_common(10)

frame = DataFrame(records)
clean_tz = frame['tz'].fillna('Missing')
tz_counts = clean_tz.value_counts()
tz_counts = tz_counts[:10]
tz_counts.plot(kind='barh')
# plt.show()

results = Series([x.split()[0] for x in frame.a.dropna()])
# print results.value_counts()[:10]
os = np.where(frame['a'].str.contains('Windows'), 'Windows', 'Not Windows')
frame['os'] = os

cframe = frame[frame.a.notnull()]
# windows = cframe['a'].str.contains('Windows')
# print len(windows)

# os = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
by_tz_os = cframe.groupby(['tz', 'os'])
agg_counts = by_tz_os.size().unstack().fillna(0)
# print agg_counts
indexer = agg_counts.sum(1).argsort()
print agg_counts.take(indexer)[-10:]
