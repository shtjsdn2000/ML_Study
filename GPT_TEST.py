import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 학습 데이터 생성
num_data = 100000
num1 = np.random.rand(num_data, 1)
num2 = np.random.rand(num_data, 1)
add_ans = num1 + num2
sub_ans = num1 - num2
mul_ans = num1 * num2
div_ans = num1 / num2

# MLP 모델 생성
model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='linear'))

# 학습 수행
model.compile(loss='mse', optimizer='adam')
model.fit(np.hstack([num1, num2]), np.hstack([add_ans, sub_ans, mul_ans, div_ans]), epochs=10, batch_size=32)

# 테스트 수행
test_data = np.array([[1, 6], [0.7, 0.8], [0.9, 0.1]])
test_result = model.predict(test_data)
print(test_result)

# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
#
# num1 = np.random.rand(100000, 1)
# num2 = np.random.rand(100000, 1)
# Ans = num1 + num2
#
# model = Sequential()
# model.add(Dense(64, input_dim=2, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='linear'))
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#
# model.fit(np.concatenate((num1, num2), axis=1), Ans, epochs=5, batch_size=64)
# result = model.predict(np.array([[0.1, 0.2]]))
# print(result)

# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# import numpy as np
#
# # 데이터를 불러옴
# fp = pd.read_excel('fp.xlsx')
# ep = pd.read_excel('ep.xlsx')
#
# # 데이터를 가공함
# #입력
# fp_test = fp.iloc[5:, 13:]
# fp_test = fp_test.reset_index(drop=True)
# fp_test.index = pd.RangeIndex(start=1, stop=len(fp_test)+1)
# fp_test.columns = ["무연탄","유류","LNG"]
# #타겟
# ep_test = ep.iloc[3:, 3:-1]
# ep_test = ep_test.reset_index(drop=True)
# ep_test.index = pd.RangeIndex(start=1, stop=len(ep_test)+1)
# ep_test.columns = ["총합"]
#
# # 가공데이터 생성및 저장함
# fp_test.to_excel(excel_writer='fp_test.xlsx')
# ep_test.to_excel(excel_writer='ep_test.xlsx')
#
# energy_input = fp_test[["무연탄","유류","LNG"]].to_numpy()
# energy_target = ep_test["총합"].to_numpy()
#
# train_input, test_input, train_target, test_target = train_test_split(energy_input,energy_target, random_state=42)
#
# # 데이터 표준화를 진행한다.
# ss = StandardScaler()
# ss.fit(train_input)
# train_scaled = ss.transform(train_input)
# test_scaled = ss.transform(test_input)
#
# # Linear Regressor 학습
# lr = LinearRegression()
# lr.fit(train_scaled, train_target)
#
# # 학습 정확도, 테스트 정확도 출력
# print("학습 정확도:", lr.score(train_scaled, train_target))
# print("테스트 정확도:", lr.score(test_scaled, test_target))
# print("정확도 차이 : ",lr.score(train_scaled, train_target)-lr.score(test_scaled, test_target))
#
# # target 데이터에서 '총합' 개수