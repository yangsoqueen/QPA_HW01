##########Data Structure
# numbers
a = 1
b = 10**3  # int
c = 1.23  # float
print (a/b)  # / between int -> int
print (a/float(b))  # / between float and int -> float
print (type(a))
print (type(c))
print (a/c)

# strings
s = 'abcdefg'
print (s)
print (s[0])
print (s[1:])
print (s[:-1])
s1 = s
s2 = 'hijklmn'
s3 = s1 + s2
print (s3)
print (s.find('a'))  # find the index of 'a'
print (s.upper())
print (s.upper().lower())

# list

# list indexing
print (a[0])  # starts from zero.
print (a[:1])
print (a[1:3])  # list[a:b] contains a, without b.
print (a[1:])
print (a[-1])

# construct a list of numbers
b = range(100)
print (b)
c = range(2, 10)
print (c)

# list operations

l1 = range(10)
l2 = range(10, 20)  # add list together
l3 = []
l3.extend(l1)  # append/extend
print (l3)  # append will contain the whole list

l3 = []
l3.append(l1)
print (l3)  # extend will only contain the elements in list

# dictionaries
dic = {'a': 1,
       'b': 2,
       'c': 3}
print (dic)
print (dic['a'])
print (len(dic))
print (dic.items())
print (dic.keys())
print (dic.values())

#############Flow control
# for statement
for k in range(10):
    print (k)

L = ['a', 'b', 'c', 'd', 'e']
for element in L:
    print (element)

# while statement
k = 0
while k < 10:
    k += 1
    print (k)

# break and continue
for k in range(100):
    print (k)
    if k > 30:
        break

for k in range(100):
    print (k)
    if k > 30:
        continue

# try/except
L = [1, 2, 3]
try:
    print (L[1])
except:
    print ('OK!')

try:
    print (L[10])
except:
    print ('Error!')


k = 0
while k < 10:
    k += 1
    try:
        print (L[k])
    except:
        print ('Error!')

###########Function
def Sum_up(x, y):
    return x + y

Prod = lambda x, y: x * y

def Fib(n):
    fib = [1, 1]
    for k in range(n):
        a = fib[-2] + fib[-1]
        fib.append(a)
    return fib

z = Sum_up(2, 3)
print (z)
y = Prod(2, 3)
print (y)
print (Fib(10))

#########I/O with Pandas
import pandas as pd
import numpy as np

# make a data frame
data = np.random.normal(0, 1, size=[1000, 3])  # generate data
df = pd.DataFrame(data, columns=['a', 'b', 'c'])  # generate a data frame
print (df)

df = pd.DataFrame({'a': data[:, 0],
                   'b': data[:, 1],
                   'c': data[:, 2]})

# Indexing
print (df['a'])
print (df['a'][2:10])
print (df[df['b'] > 0])
print (df[df['b'] > 0][df['c'] < 0])

# save data frame
df.to_csv('F:/random_matrix.csv', index=True)

# read from csv file
df = pd.read_csv('F:/random_matrix.csv')
print (df)

# same directory as the py file
df.to_csv('./random_matrix.csv', index=True)
df = pd.read_csv('./random_matrix.csv', index_col=0)
print (df)

#########numpy
import numpy as np
A = np.array(range(100))
print (A)
print (A[0])
print (A.reshape(20, 5))
A = np.array([[1, 2, 3], [3, 4, 5]])
print (A)

A = np.zeros([10, 5])
print (A)
A = np.random.normal(loc=0, scale=1, size=[10, 5])  # generate random matrix
print (A)

# Operations
print (np.size(A, axis=0))
print (np.sum(A))
print (np.std(A))
print (np.var(A))
print (np.average(A))
print (np.sort(A))
print (A.transpose())  # transpose for matrix

# matrix multiplication
A = np.array([[1, 2, 3], [3, 4, 5]])
print (A*A)
print (np.dot(A, A.transpose()))  #For 2-D arrays it is equivalent to matrix multiplication, and for 1-D arrays to inner product of vectors

###########scipy
from scipy.stats import norm, beta
from matplotlib import pyplot as plt
import numpy as np
print (norm.pdf(x=1))
print (norm.cdf(x=1))
print (beta.cdf(x=0.6, a=1, b=2))
print (beta.pdf(x=0.4, a=13, b=4))

# plot beta pdf and cdf
X = np.linspace(0, 1, 1000)
Y = beta.pdf(X, a=13, b=5)
Z = beta.cdf(X, a=13, b=5)
plt.plot(X, Y)
plt.plot(X, Z)

