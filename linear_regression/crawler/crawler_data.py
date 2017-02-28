# coding=utf-8

print "aa"

a = 100
if a >= 0:
    print a
else:
    print -a

a = True
a = False
a = None
print a

a = [1, 2, 3, 4]
b = a
a.append(5)
print b

print isinstance(a, int)

tiny_list = ['aa', 'bb', 'cc', 'dd', 'ee']

print tiny_list[1:3]

dict_a = {'one': 'name'}
print dict_a.keys()

a = 'foo'

def attempt_float(x1):
    try:
        return float(x1)
    except (TypeError, ValueError):
        return x1
    finally:
        return x1

print attempt_float(1.12)

x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
for a, b, c in zip(x, y, z):
    print str(a) + ":" + str(b) + ":" + str(c)






