# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:08:56 2019

@author: Tobias
"""

start     = dt.datetime(1975, 1, 1)
end       = dt.datetime(2018,1,1)
c         = web.get_data_fred('PCE',start=start,end=end)
c_growth  = np.log(c['PCE'])-np.log(c['PCE'].shift(1))   
sp_500    = web.get_data_yahoo(['^GSPC'],start=start,end=end)
sp_growth = np.log(sp_500['Close'])-np.log(sp_500['Close'].shift(1)) 
tbill     = web.get_data_fred('TB3MS',start=start,end=end)

beta  = 0.99
stdev = np.empty((0,0))
for rho in range(1,26):
    m     = beta*(1+c_growth)**(-rho)
    stdev = np.append(stdev,np.std(m))

fig = plt.figure()
plt.plot(stdev)
fig.suptitle('M', fontsize=20)
plt.xlabel('rho', fontsize=18)
plt.ylabel('sigma', fontsize=16)

# Find the Sharpe ratio
ER = np.mean(sp_growth)