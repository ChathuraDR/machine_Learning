import pandas as pd 
import quandl
import math

df = quandl.get("WIKI/GOOGL", authtoken="Wb5tSw5Tuqv7ZyMnYZzm")

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/ df['Adj. Close'] * 100.0
df['PCT_chanage'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100.0 

df = df[['Adj. Close','HL_PCT','PCT_chanage','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) #replace empty data with this

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True )
print(df.head())
 