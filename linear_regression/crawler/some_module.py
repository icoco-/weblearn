# coding=utf-8

import json
from collections import defaultdict
from collections import Counter
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# top k 不同实现方法

path = './usagov_bitly_data.txt'

records = [json.loads(line, encoding='utf-8') for line in open(path)]

time_zones = [rec['tz'] for rec in records if 'tz' in rec]


def get_zones_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1

    return counts


def get_zones_counts2(sequence):
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts


def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

zones_counts = get_zones_counts(time_zones)

# print top_counts(zones_counts)

# use counter
counts = Counter(zones_counts)
# print counts.most_common(10)

frame = DataFrame(records)
tz_counts = frame['tz'].value_counts()
# print tz_counts[:10]

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
# tz_counts[:10].plot(kind='barh', rot=0)
# plt.show()

# top agent
results = Series([x.split()[0] for x in frame.a.dropna()])

# windows un_windows user
c_frame = frame[frame.a.notnull()]
operating_system = np.where(c_frame['a'].str.contains('Windows'),
                            'Windows', 'Not Windows')
by_tz_os = c_frame.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
print agg_counts[:10]

indexer = agg_counts.sum(1).argsort()
count_subset = agg_counts.take(indexer)[-10:]
# count_subset.plot(kind='barh', stacked=True)
normed_subset = count_subset.div(count_subset.sum(1), axis=0)
normed_subset.plot(kind='barh', stacked=True)
plt.show()


