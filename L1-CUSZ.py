from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, gaussian_kde, t, laplace
import collections

#read data

dy = pd.read_excel('./700HK.xlsx', index_col=0)
dy.index = pd.to_datetime(dy.index)
dyRet = dy.pct_change().dropna() #np.log(dy / dy.shift()).dropna()
mu = dyRet.mean()[0]
sig = dyRet.std()[0]

yRet = np.array(dyRet)

plt.subplot(221)
plt.hist(yRet, normed=True, bins=100, color='grey')
distance = np.linspace(min(yRet),max(yRet))
plt.plot(distance, norm.pdf(distance,mu,sig), c='r')
plt.xlabel('log return')
plt.ylabel('density')
plt.legend(loc="upper right", fontsize=5)

plt.subplot(222)
yNRet = (yRet-mu)/sig #standardization
distanceN = np.squeeze(np.linspace(min(yNRet), max(yNRet)))
kernel = gaussian_kde(np.squeeze(yNRet))
plt.plot(distanceN, norm.pdf(distanceN,0,1), label='Normal', c='r')
plt.plot(distanceN, kernel(distanceN), label='empirical', c='grey')
plt.legend(loc="upper right", fontsize=5)

plt.subplot(223)
plt.plot(distanceN, norm.pdf(distanceN,0,1), label='Normal', c='r')
plt.plot(distanceN, t.pdf(distanceN, df=2), label='t-dist, df=2', c='g')
plt.plot(distanceN, kernel(distanceN), label='empirical', c='grey')
plt.legend(loc="upper right", fontsize=5)

plt.subplot(224)
plt.plot(distanceN, norm.pdf(distanceN,0,1), label='Normal', c='r')
plt.plot(distanceN, kernel(distanceN), label='empirical', c='grey')
plt.plot(distanceN, t.pdf(distanceN, df=2), label='t-dist, df=2', c='g')
plt.plot(distanceN, laplace.pdf(distanceN), label='laplace-dist', c='y')
plt.legend(loc="upper right", fontsize=5)


############################################
#create an empty dictionary
v = {}
v = collections.OrderedDict()
year = np.unique(dy.index.year)
month = np.unique(dy.index.month)

for yi in year:
    for mi in month:
        try:
            temp = dyRet.loc[(dyRet.index.year == yi) & (dyRet.index.month == mi)]
            if len(temp)==0:
                break
            else:
                v[yi *100+ mi] = []
                vol = np.std(temp)
            v[yi *100+ mi].extend(vol)

        except:
            continue

pd.DataFrame.from_dict(v, orient='index').to_csv('./rollingSig.csv')


############### resample ##########
dyRet.resample('m').std()