# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 08:34:05 2019

@author:  ~ 
"""

import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pickle
import requests
import bs4 as bs
import sys


tickers    = ['AAPL','MSFT','GOOGL','GLD']
data       = pd.DataFrame()
start      = dt.datetime(2005, 1, 1)
end        = dt.datetime(2018,1,1)
for ticker in tickers:
    data[ticker] = web.get_data_yahoo(ticker,start=start,end=end)['Close']
    
data.columns = tickers
ret_daily    = np.log(data / data.shift(1))
ret_mean_an  = ret_daily.mean()*252
vcov         = ret_daily.cov()*252
print('.'*100)
print(' \n Annualized mean returns: \n')
print(ret_mean_an)
print('\n Covariance matrix of returns: \n')
print(vcov)
print('.'*100)


port_std = []
port_ret = []
simLen   = 10000
print('\nProgress:')
print('.'*100 + '\n')
for ii in range(1,simLen+1):
    weights   = np.random.normal(0,1,len(tickers))
    weights  /= weights.sum()
    if np.any(weights>=2) or np.any(weights<=-2): continue
    temp      = weights*vcov*np.transpose(weights)
    port_std  = np.append(port_std,sum(temp.sum())**0.5)
    port_ret  = np.append(port_ret,sum(weights*ret_mean_an))
    if (ii % 100==0):
        b=('Finished with iteration ' + str(ii) + ' of ' + str(len(range(1,simLen+1))))
        sys.stdout.write('\r'+b)

## Identify mean-variance portfolio
risk_free   = web.get_data_fred('TB3MS',start=end,end=end)/100
s_p         = (port_ret-risk_free['TB3MS'][0])/port_std
s_p_m_ind   = np.argmax(s_p)
min_var_ind = np.argmin(port_std)



fig, ax = plt.subplots()
ax.scatter(port_std, port_ret, c='blue')
ax.scatter(port_std[s_p_m_ind], port_ret[s_p_m_ind], c='red',marker='D')
ax.scatter(port_std[min_var_ind], port_ret[min_var_ind], c='red',marker='*')
ax.plot([0,port_std[s_p_m_ind]], [risk_free['TB3MS'][0],port_ret[s_p_m_ind]],c='r')
ax.plot([port_std[s_p_m_ind],port_std[s_p_m_ind]*3], [port_ret[s_p_m_ind],(port_ret[s_p_m_ind]-risk_free['TB3MS'][0])*3],c='r',linestyle='--')
plt.xlim(left=0)
plt.title("The envelope")
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$R_p$')
plt.show()




### What if we add more assets? (Seidel) ###

def get_sp500_tickers():
    resp    = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup    = bs.BeautifulSoup(resp.text, 'lxml')
    table   = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers

sp_tickers = get_sp500_tickers() 

## Fetch 20 random tickers ##
magic_indicies = np.random.randint(1,500,20)
ext_tickers    = tickers
for ind in magic_indicies: 
    ext_tickers = np.append(ext_tickers,sp_tickers[ind])



