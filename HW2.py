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
tbillog   = np.log(tbill/100+1)

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
sp_500_m    = sp_500.asfreq('M',method='bfill')
sp_growth_m = np.log(sp_500_m['Close'])-np.log(sp_500_m['Close'].shift(1)) 



#Er       = np.mean(sp_growth)*21 # Assuming 21 trading days on average per month
Er       = np.mean(sp_growth_m)
sigma_r  = np.std(sp_growth_m)
rf       = np.mean(tbillog)/12
R        = Er-rf[0]
SR       = R/sigma_r


