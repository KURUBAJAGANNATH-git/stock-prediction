import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.api.models import load_model
import streamlit as st 
import matplotlib.pyplot as plt

model = load_model("C:\\Users\\jkuruba\\Stock Prediction Model.keras")

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train=pd.DataFrame(data.Close[0 : int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80) : len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pass_100_data = data_train.tail(100)
data_test= pd.concat([pass_100_data, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs Moving Average for 50 Days')
ma_50days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,8))
plt.plot(ma_50days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA for 50 Days vs MA for 100 Days')
ma_50days = data.Close.rolling(50).mean()
ma_100days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(10,8))
plt.plot(ma_50days, 'r')
plt.plot(ma_100days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

y_predict = model.predict(x)

scale = 1/scaler.scale_

y_predict = y_predict*scale
y = y*scale

st.subheader('Actual Price vs Predicted Price')
fig3 = plt.figure(figsize=(10,8))
plt.plot(y_predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig3)