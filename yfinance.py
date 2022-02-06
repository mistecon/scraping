#Stock data from Yahoo! Finance
#How should we deal with the data points with the jump in time? (ex. market is closing...)

from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import math

#stocks = pd.read_csv("C:/Users/Hideto Kamei/Documents/04_Programming/01_Scraping/yfinance/code_list.csv")
stocks = pd.read_csv("C:/Users/Hideto Kamei/Documents/04_Programming/01_Scraping/yfinance/NASDAQ_traded.csv")

#periodicity; daily, weekly, monthly, intraday
#interval; if periodicity is intraday; "1min", "5min", "15min", "30min", or "60min" 
time_int = '1min'
o_price = pd.DataFrame()
vol = pd.DataFrame()

#for code in codes["Symbol"]:
for code in stocks["Symbol"]:
    # initialize TS object with API key and output format
    ts = TimeSeries(key='Your-API-Key', output_format='pandas')
    # Get the data
    #data, meta_data = ts.get_daily(symbol='D', outputsize='full')
    try:
        data, meta_data = ts.get_intraday(symbol=code, interval=time_int, outputsize='full')
        o_price[code] = data["1. open"]
        vol[code] = data["5. volume"]        
    except:
        pass
    # Print the data
    print(code)
    print(data.head())
o_price.to_csv("C:/Users/Hideto Kamei/Documents/04_Programming/01_Scraping/yfinance/stock_1min_oprice.csv",encoding='UTF-8')
vol.to_csv("C:/Users/Hideto Kamei/Documents/04_Programming/01_Scraping/yfinance/stock_1min_vol.csv",encoding='UTF-8')
#check the number of nans for each column

res = pd.read_csv("C:/Users/Hideto Kamei/Documents/04_Programming/01_Scraping/yfinance/stock_1min.csv")
#select columns which have more than 90% of datapoints filled
cond = pd.Series(res.isnull().sum()<=res.shape[0]*0.1)
res = res.ix[:,cond]
res = res.ix[:,1:]

def change(res):
    res_prev = res.iloc[0:res.shape[0]-1,:]
    res_post = res.iloc[1:res.shape[0],:]
    res_prev.index = res_post.index
    change = np.log(res_post)-np.log(res_prev)
    return(change)

#compute the correlation of normalized G
del_t=100
def g_series(change,del_t):
    g=pd.DataFrame(data=None,index=change.index,columns=change.columns)
    #g["date"] = change["date"]
    for t in range(0,change.shape[0]-del_t):
        #take the sample from time t to t+del_t, omitting "date" column
        window_sample = change.iloc[t:t+del_t,]
        window_ave = window_sample.mean()
        window_std = window_sample.std()
        g.iloc[t+int(del_t/2),] = (change.iloc[t+int(del_t/2),]-window_ave)/window_std
    g = g[int(del_t/2):change.shape[0]-int(del_t/2)]
    return(g)

#check for nan
#for i in range(0,g.shape[0]):
#    if g.isnull().all(axis=1)==True:
#        print(i)

def corr(g):
    corr = pd.DataFrame(data=None, index=g.columns, columns=g.columns)
    for i in g.columns:
        for j in g.columns:
            gij = g[[i,j]]
            gij = gij[gij.isnull().any(axis=1)==False]
            corr.loc[i,j] = np.dot(gij.iloc[:,0],gij.iloc[:,1])
    return(corr)
    
change = change(res)
T = change.shape[0]
g = g_series(change,del_t)
corr = corr(g)/T
w,v = LA.eig(np.array(corr,dtype=float))
plt.hist(w,bins=40,range=(0,2))

pd.Series(w).to_csv("C:/Users/Hideto Kamei/Documents/04_Programming/01_Scraping/yfinance/evalue.csv",encoding='UTF-8')

#compute the seperation of the spectrum
dw = np.diff(np.sort(w))
plt.hist(dw,bins=30,range=(0,0.1))

#Plot the histogram for the rate of change
plt.hist(change["A"],bins=30,log=True,range=(-0.003,0.003))
plt.hist(change["AKS"],bins=30,log=True,range=(-0.02,0.02))
plt.hist(change["PKG"],bins=30,log=True,range=(-0.003,0.003))

#Plot spectral form factor
#x = np.linspace(0, 10000, 10000)
lnx = np.linspace(0, np.log(10000), 10000)
x = np.exp(lnx)
#test=np.array([1,1.1,1.21])
res=np.zeros(len(x))
for i in range(0,len(x)):
    res[i] = abs(sum(np.exp(x[i]*w*1j)))
plt.plot(np.log(x),np.log(res))

plt.plot(x, abs(np.exp(x*1j)+np.exp(1.1*x*1j)) )
