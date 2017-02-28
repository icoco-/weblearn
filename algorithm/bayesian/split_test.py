# coding=utf-8

import re
from numpy import *
import feedparser

mySent = 'This book is the best book on Python or M.L. I have ever laid  eyes upon.'
listOfTokens = mySent.split("\\W*")  # 去掉单词中的标点符号，分隔符是除单词、数字外的所有字符串
# print [tok.lower() for tok in listOfTokens if len(tok) > 0]

emailTest = open('data/email/ham/6.txt').read()
# print emailTest.split()

a = [1, 2, 3]
b = [4, 5, 6]
a.append(b)
# print a
a.extend(b)
# print a

# print int(random.uniform(0, 50))

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
print len(ny['entries'])

