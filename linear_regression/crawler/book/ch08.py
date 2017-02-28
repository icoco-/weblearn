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
import matplotlib.pyplot as plt
from datetime import datetime

# k-- 黑色虚线
# plt.plot(randn(50), 'k--')
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
#
# plt.plot(randn(50).cumsum(), 'k--')
# _ = ax1.hist(randn(100), bins=20, color='k', alpha=0.3)
# ax2.scatter(np.arange(30), np.arange(30) + 3 * randn(30))

# fig, axes = plt.subplots(2, 3)
#
# plt.show()

# plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                 wspace=None, hspace=None)
# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
# for i in range(2):
#     for j in range(2):
#         axes[i, j].hist(randn(500), bins=50, color='k', alpha=0.5)
# plt.subplots_adjust(wspace=0, hspace=0, left=None, bottom=None, right=None, top=None)

# plt.figure()
# plt.plot(randn(30).cumsum(), 'ko--')
# data = randn(30).cumsum()
# plt.plot(data, 'k--', label='Default')
# plt.plot(data, 'k-', drawstyle='steps-post', label='steps-post')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(randn(1000).cumsum())
#
# ticks = ax.set_xticks([0, 250, 500, 750, 1000])
# labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
#                             rotation=30, fontsize='small')
# ax.set_title('My first matplotlib plot')
# ax.set_xlabel('Stages')
# plt.show()

# fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
# ax.plot(randn(1000).cumsum(), 'k', label='one')
# ax.plot(randn(1000).cumsum(), 'k--', label='two')
# ax.plot(randn(1000).cumsum(), 'k.', label='three')
#
# ax.legend(loc='best')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# data = pd.read_csv('ch06/spx.csv', index_col=0, parse_dates=True)
# spx = data['SPX']
#
# spx.plot(ax=ax, style='k-')
#
# crisis_data = [
#     (datetime(2007, 10, 11), 'Peak of bull market'),
#     (datetime(2008, 3, 12), 'Bear Stearns Fails'),
#     (datetime(2008, 9, 15), 'Lehman Bankruptcy')
# ]
#
# for date, label in crisis_data:
#     ax.annotate(label, xy=(date, spx.asof(date) + 50),
#                 xytext=(date, spx.asof(date) + 200),
#                 arrowprops=dict(facecolor='black'),
#                 horizontalalignment='left', verticalalignment='top')
#
# # Zoom in on 2007-2010
# ax.set_xlim(['1/1/2007', '1/1/2011'])
# ax.set_ylim([600, 1800])
#
# ax.set_title('Important dates in 2008-2009 financial crisis')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
# circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
# pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
#                    color='g', alpha=0.5)
#
# ax.add_patch(rect)
# ax.add_patch(circ)
# ax.add_patch(pgon)
# plt.savefig('figpath.pdf')

# s = Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
# print s
# s.plot()
# plt.show()

# df = DataFrame(np.random.randn(10, 4).cumsum(0),
#                columns=['A', 'B', 'C', 'D'],
#                index=np.arange(0, 100, 10))
# df.plot()
# plt.show()

# fig, axes = plt.subplots(1, 2)
# data = Series(np.random.rand(16), index=list('abcdefghijklmnop'))
# data.plot(kind='bar', ax=axes[0], color='k', alpha=0.7)
# data.plot(kind='barh', ax=axes[1], color='k', alpha=0.7)
# plt.show()

# df = DataFrame(np.random.rand(6, 4),
#                index=['one', 'two', 'three', 'four', 'five', 'six'],
#                columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
# df.plot(kind='bar', stacked=True, alpha=0.5)
# plt.show()

tips = pd.read_csv('ch06/tips.csv')
# party_counts = pd.crosstab(tips.size, tips.day)
party_counts = tips.groupby(['day', 'size']).size().unstack().fillna(0)

party_counts = party_counts.ix[:, 2:5]
print party_counts
party_pcts = party_counts.div(party_counts.sum(1).astype(float), axis=0)
# print party_pcts
party_pcts.plot(kind='bar', stacked=True)
plt.show()


