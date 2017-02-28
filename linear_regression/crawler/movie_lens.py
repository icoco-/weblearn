# coding=utf-8

import pandas as pd
import numpy as np

u_names = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('movie_data/users.dat', sep='::', header=None, names=u_names, engine='python')
# print users

r_names = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('movie_data/ratings.dat', sep='::', header=None, names=r_names, engine='python')

m_names = ['movie_id', 'title', 'genres']
movies = pd.read_table('movie_data/movies.dat', sep='::', header=None, names=m_names, engine='python')

all_data = pd.merge(pd.merge(ratings, users), movies)
# print all_data.ix[np.arange(5), ['user_id', 'rating', 'title']]

# select title, avg(rating) group by gender, title
mean_ratings = all_data.pivot_table('rating', index=['title'], columns='gender', aggfunc='mean')
# print mean_ratings[:5]

# select movie's ratings num >= 250
ratings_by_title = all_data.groupby('title').size()
active_titles = ratings_by_title.index[ratings_by_title >= 250]
# print active_titles[:5]

# female most like movie
mean_ratings = mean_ratings.ix[active_titles]
top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)
# print top_female_ratings[:5]

# biggest diff between male and female
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff')

# arr[start:end:step] -1 respect reorder
sorted_by_diff = sorted_by_diff[::-1][:15]
# print sorted_by_diff[:15]

rating_std_by_title = all_data.groupby('title')['rating'].std()
rating_std_by_title = rating_std_by_title.ix[active_titles]
# print rating_std_by_title.sort_values(ascending=False)[:10]

all_ratings_num = all_data.groupby('title').size()
print all_ratings_num.sort_values(ascending=False)[:15]





