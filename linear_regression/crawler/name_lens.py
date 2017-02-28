# coding=utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# names2010 = pd.read_csv('names/yob2010.csv', names=['name', 'sex', 'births'])
# print names2010[:10]
#
# print names2010.groupby('sex').size()

years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    path = 'names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)

    frame['year'] = year
    pieces.append(frame)

# merge all pieces into names
names = pd.concat(pieces, ignore_index=True)
# print names
total_births = names.pivot_table('births', index='year', columns='sex', aggfunc=sum)
# print total_births.tail(10)


def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births / births.sum()
    return group
names = names.groupby(['year', 'sex']).apply(add_prop)
# print names.groupby(['year', 'sex']).sum()[:15]
# print names[:15]
# check sum equals 1
# print np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)


def get_top1000(group):
    return group.sort_values(by='births', ascending=False)[:1000]

grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)
# print top1000[:100]
#
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
print boys[:10]


# # 名字的数量变化
# # total_births = top1000.pivot_table('births', index='year', columns='name', aggfunc=sum)
# # subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
#
# # subset.plot(subplots=True, figsize=(12, 10), grid=True, title='Number of births per year')
# # plt.show()
#
# # 最常见的1000个名字所占的比例
# # table = top1000.pivot_table('prop', index='year', columns='sex', aggfunc=sum)
# # table.plot(title='Sum of table1000.prop by year and sex', yticks=np.linspace(0, 1.2, 13),
# #            xticks=range(1880, 2020, 10))
# # plt.show()
#
# # 占出生人数前50%的不同名字的数量
# # df = boys[boys.year == 2010]
# # prop_cum_sum = df.sort_index(by='prop', ascending=False).prop.cumsum()
# # 先计算累计和，然后二分查找插入位置 2010:117,  1900:25
# # print prop_cum_sum.searchsorted(0.5)
#
#
# def get_quantity_count(group, q=0.5):
#     group = group.sort_index(by='prop', ascending=False)
#     return group.prop.cumsum().searchsorted(q) + 1
#
# # diversity = top1000.groupby(['year', 'sex']).apply(get_quantity_count)
# # diversity = diversity.unstack('sex')
# # print diversity[:10]
# # diversity.plot(title="number")
# # plt.show()
#
# last_letters = names.name.map(lambda x: x[-1])
# last_letters.name = 'last_letter'
# table = names.pivot_table('births', index=last_letters, columns=['sex', 'year'], aggfunc=sum)
# sub_table = table.reindex(columns=[1910, 1960, 2010], level='year')
# letter_prop = sub_table / sub_table.sum().astype(float)
#
# # fig, axes = plt.subplots(2, 1, figsize=(10, 8))
# # letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
# # letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female', legend=False)
# # plt.show()
#
# all_names = top1000.name.unique()
# mask = np.array(['lesl' in x.lower() for x in all_names])
# lesley_like = all_names[mask]
# filtered = top1000[top1000.name.isin(lesley_like)]
# table = filtered.pivot_table('births', index='year', columns='sex', aggfunc='sum')
# table = table.div(table.sum(1), axis=0)
# table.plot(style={'M': 'k-', 'F': 'k--'})
# plt.show()






