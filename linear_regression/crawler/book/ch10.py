# coding=utf-8

from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import pytz
from numpy.random import randn
import numpy as np
pd.options.display.max_rows = 12
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 4))
from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse
from pandas.tseries.offsets import Hour, Minute, Day, MonthEnd

now = datetime.now()
# print now.year, now.month, now.day

delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
# print delta.days, delta.seconds
start = datetime(2011, 1, 7)
# print start + timedelta(12)

stamp = datetime(2011, 1, 3)
# print str(stamp)
# print stamp.strftime('%Y-%m-%d')
value = '2011-01-03'
# print datetime.strptime(value, '%Y-%m-%d')
datestrs = ['7/6/2011', '30/6/2011']
# print [datetime.strptime(x, '%m/%d/%Y') for x in datestrs]

# print parse('2011/01/03')
# print parse('Jan 31, 1997 10:45 PM')

idx = pd.to_datetime(datestrs + [None])
# print pd.isnull(idx)

dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
         datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = Series(np.random.randn(6), index=dates)
# print type(ts)
# print ts['2011-01-02']

longer_ts = Series(np.random.randn(1000),
                   index=pd.date_range('1/1/2000', periods=1000))
# print longer_ts[:datetime(2000, 1, 5)]

dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
long_df = DataFrame(np.random.randn(100, 4),
                    index=dates,
                    columns=['Colorado', 'Texas', 'New York', 'Ohio'])
# print long_df.ix['5-2001']
# print pd.date_range('1/1/2000', '5/1/2000', freq='BM')
# print pd.date_range('5/2/2012 12:56:31', periods=5, normalize=True)

# print pd.date_range('1/1/2000', '1/3/2000 23:59', freq='4h')
# print pd.date_range('1/1/2000', periods=10, freq='1h30min')

ts = Series(np.random.randn(4),
            index=pd.date_range('1/1/2000', periods=4, freq='M'))
# print ts.shift(1) - 1

# print pytz.common_timezones[-5:]

rng = pd.period_range('1/1/2000', '6/30/2000', freq='M')
# print rng
rng = pd.period_range('2010Q3', '2012Q4', freq='Q-JAN')
rng = pd.date_range('1/1/2000', periods=3, freq='M')
ts = Series(randn(3), index=rng)
pts = ts.to_period()
# print pts

data = pd.read_csv('ch06/macrodata.csv')
index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
data.index = index
# print data.infl

rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = Series(randn(len(rng)), index=rng)
# print ts
# print ts.resample('M', how='mean')

rng = pd.date_range('1/1/2000', periods=12, freq='T')
ts = Series(np.arange(12), index=rng)
# print ts.resample('5min', closed='left', label='left').sum()
# print ts.resample('5min').ohlc()

rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = Series(np.arange(100), index=rng)
# print ts.groupby(lambda x: x.month).mean()
# print ts.groupby(lambda x: x.weekday).mean()

frame = DataFrame(np.random.randn(2, 4),
                  index=pd.date_range('1/1/2000', periods=2, freq='W-WED'),
                  columns=['Colorado', 'Texas', 'New York', 'Ohio'])
df_daily = frame.resample('D')
# frame.resample('D', fill_method='ffill')

close_px_all = pd.read_csv('ch06/stock_px.csv', parse_dates=True, index_col=0)
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
close_px = close_px.resample('B').ffill()
# close_px['AAPL'].plot()
# close_px.ix['2009'].plot()
appl_q = close_px['AAPL'].resample('Q-DEC').ffill()
appl_q.ix['2009'].plot()
plt.show()



















