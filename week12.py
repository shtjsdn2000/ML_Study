import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout
#통계적으로 주가예측은 불가
#but 개별 종목은 가능하지않을까...? //개인적으로 확인하는 기술을 알려주지 않음
#그래서 알아내는 방법을 머신러닝으로 알아보자
#(시가:그날처음거래한 가격(전날 종가랑 같을수도 다를수도 있음), 종가(마지막10분동안 접수받은가격),고가 ,저가 ,거래량)
#2주전거래 데이터 ~ 어제 거래데이터 -->오늘종가로 mapping
#과거 데이터와 연관관계를 LSTM으로 알아봄

#데이터 로드
#005930 삼성전자.
df = fdr.DataReader('005930', '2018-05-04', '2023-07-18') #005930 삼성전자.
print(df.shape)
#종목코드와 관찰구간은 적절히 설정
#df는 dictionar로 되어있음. dfkey() 출력해 볼 것
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

dfx = df[['Open','High','Low','Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]
dfx.describe() #모든 값이 0과 1 사이인 것 확인

X = dfx.values.tolist()
y = dfy.values.tolist()

window_size = 10
data_X = []
data_y = []
for i in range(len(y) - window_size):
    _X = X[i : i + window_size]
    _y = y[i + window_size]
    data_X.append(_X)
    data_y.append(_y)
print(_X, "->", _y)

print('전체 데이터의 크기 :', len(data_X), len(data_y))
train_size = int(len(data_y) * 0.7)
train_X = np.array(data_X[0 : train_size])
train_y = np.array(data_y[0 : train_size])
test_size = len(data_y) - train_size
test_X = np.array(data_X[train_size : len(data_X)])
test_y = np.array(data_y[train_size : len(data_y)])
print('훈련 데이터의 크기 :', train_X.shape, train_y.shape)
print('테스트 데이터의 크기 :', test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(units=20, activation='relu', return_sequences=True, input_shape=(10, 4)))
model.add(Dropout(0.1))
model.add(LSTM(units=20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_X, train_y, epochs=30, batch_size=1)
pred_y =model.predict(test_X)

pred_y = model.predict(test_X)
plt.figure()
plt.plot(test_y, color='red', label='real SEC stock price')
plt.plot(pred_y, color='blue', label='predicted SEC stock price')
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()
print("내일 SEC 주가 :", df.Close[-1] * pred_y[-1] / dfy.Close[-1], 'KRW')