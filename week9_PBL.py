# 4칙연산 계산기
# Multi-Layer Perceptron 활용
# • 덧셈 계산기 만들기
# • 덧셈 학습 데이터를 100000개 만드세요.
# • num1: 100000행 1열 랜덤 넘버
# • Num2: 100000행 1열 랜덤 넘버
# • Ans: 100000행 1열 (num1과 num2를 더한 결과)
# • MLP Layer를 구성
# • 사용할 Layer 수 결정
# • Activation Function 결정 # --> 종류에 따라 학습이 되기도 하고 안되기도 함
# • 결과 관찰  계산 정확도를 높이려면?
#은닉층은 2개가 적당함
#코드 로는 11줄 정도임
#입력 숫자2개 출력은 하나


import tensorflow as tf
from tensorflow import keras
import numpy as np

#
data_size = 100000 #학습데이터 수
# 학습 범위를 0 ~ 200 까지 발생 시킴 //값을 키울수록 똑똑해지지만 오래걸림
#train_data = np.random.randin(100,size=(data_size,2)) # 0~100 까지를 2 개 더한다.
train_data = np.random.randint(100,size=(data_size,2)) #-50 ~ +50 범위 출력값이 - 가 나올 수 도 있는상황
train_ans_add=(train_data[:,0]+train_data[:,1])

print("덧셈",np.shape(train_ans_add))

# 히든 레이어 3개 // 출력은 1개 여야함
# 뺼셈은 어떤 함수를 써야 하는지 고민해봐야함
#모델을 몇개쓸지는 본인 마음이지만 1개로 하는건 상당히 어려움 즉, 그냥 여러개 쓰자
model = keras.Sequential() # 학습데이터의 형태 및 활성화함수의 종류를 고민해봐야함
model.add(keras.layers.Dense(10,activation='relu'))#중간에 시그모이드는 상관없음 근데 학습이 좀 느림 결과도 부정확함
model.add(keras.layers.Dense(20,activation='relu')) #레이어는 (1~2)개만 해도 충분한 갯수는 크게 상관없음
model.add(keras.layers.Dense(5,activation='relu')) #relu는 "-" 표현 안됨 -일때는 고민해 봐야함
model.add(keras.layers.Dense(1,activation='elu')) #시그모이드는  0~1이기에 쓸수없음
#mse? 5+5 예측12 정답은 10 mse --> 4 예측 - 정답 ^ 2 = mse
model.compile(loss='mse',optimizer=tf.optimizers.Adam(0.00029), metrics=['accuracy'])
model.fit(train_data, train_ans_add,batch_size=100,epochs=10)


print("덧셈")
z= np.array([29.1,32.5]).reshape(1,2)
q=model.predict(z,batch_size=1)
print("첫번째",q)

z= np.array([30.2,2]).reshape(1,2)
q=model.predict(z,batch_size=1)
print("두번째",q)

z= np.array([100,100]).reshape(1,2)
q=model.predict(z,batch_size=1)
print("세번째",q)
#


#뺄셈
#--------------------------------------------------------------------------------------------------------------------
#
#
data_size = 100000 #학습데이터 수
# 학습 범위를 0 ~ 200 까지 발생 시킴 //값을 키울수록 똑똑해지지만 오래걸림
#train_data = np.random.randin(100,size=(data_size,2)) # 0~100 까지를 2 개 더한다.
train_data = np.random.randint(200,size=(data_size,2)) - 100 #-50 ~ +50 범위 출력값이 - 가 나올 수 도 있는상황
train_ans_sub=(train_data[:,0]-train_data[:,1])

print("뺄셈",np.shape(train_ans_sub))

# 히든 레이어 3개 // 출력은 1개 여야함
# 뺼셈은 어떤 함수를 써야 하는지 고민해봐야함
#모델을 몇개쓸지는 본인 마음이지만 1개로 하는건 상당히 어려움 즉, 그냥 여러개 쓰자
model = keras.Sequential() # 학습데이터의 형태 및 활성화함수의 종류를 고민해봐야함
model.add(keras.layers.Dense(10,activation='tanh'))#중간에 시그모이드는 상관없음 근데 학습이 좀 느림 결과도 부정확함
#model.add(keras.layers.Dense(20,activation='tanh')) #레이어는 (1~2)개만 해도 충분한 갯수는 크게 상관없음
model.add(keras.layers.Dense(5,activation='tanh')) #relu는 "-" 표현 안됨 -일때는 고민해 봐야함
model.add(keras.layers.Dense(1,activation='elu')) #시그모이드는  0~1이기에 쓸수없음
#mse? 5+5 예측12 정답은 10 mse --> 4 예측 - 정답 ^ 2 = mse
model.compile(loss='mse',optimizer=tf.optimizers.Adam(0.00029), metrics=['accuracy'])
model.fit(train_data, train_ans_sub,batch_size=100,epochs=30)


#문제 10.1 + 20.3 해봐라 소수점은 가르치지 않았음 따라서 소수점은 빼고 계산해서 똑같은 결과가 나옴
print("뺼셈")
z= np.array([32.5,29.1]).reshape(1,2)
q=model.predict(z,batch_size=32)
print("첫번째 답 : 3.4",q)

z= np.array([30,2]).reshape(1,2)
q=model.predict(z,batch_size=32)
print("두번째 답 : 28",q)

z= np.array([20,10]).reshape(1,2)
q=model.predict(z,batch_size=32)
print("세번째 : 10 ",q)

# #곱셈
# #--------------------------------------------------------------------------------------------------------------------


data_size = 100000 #학습데이터 수
# 학습 범위를 0 ~ 200 까지 발생 시킴 //값을 키울수록 똑똑해지지만 오래걸림
train_data = np.random.randint(100,size=(data_size,2)) # 0~100 까지를 2 개 더한다.
#train_data = np.random.randint(100,size=(data_size,2)) + np.random.rand(data_size, 2) # + np.random.rand(data_size, 2)  #-50 ~ +50 범위 출력값이 - 가 나올 수 도 있는상황
train_ans_mul=(train_data[:,0] * train_data[:,1])
print("input_data: ",train_data)
print("곱셈",np.shape(train_ans_mul))

# 히든 레이어 3개 // 출력은 1개 여야함
# 뺼셈은 어떤 함수를 써야 하는지 고민해봐야함
#모델을 몇개쓸지는 본인 마음이지만 1개로 하는건 상당히 어려움 즉, 그냥 여러개 쓰자
model = keras.Sequential() # 학습데이터의 형태 및 활성화함수의 종류를 고민해봐야함
model.add(keras.layers.Dense(10,activation='tanh'))#중간에 시그모이드는 상관없음 근데 학습이 좀 느림 결과도 부정확함
model.add(keras.layers.Dense(20,activation='tanh')) #레이어는 (1~2)개만 해도 충분한 갯수는 크게 상관없음
# model.add(keras.layers.Dense(5,activation='relu')) #relu는 "-" 표현 안됨 -일때는 고민해 봐야함
model.add(keras.layers.Dense(1,activation='elu')) #시그모이드는  0~1이기에 쓸수없음
#mse? 5+5 예측12 정답은 10 mse --> 4 예측 - 정답 ^ 2 = mse
model.compile(loss='mse',optimizer=tf.optimizers.Adam(0.00029), metrics=['accuracy'])
model.fit(train_data, train_ans_mul,batch_size=100,epochs=30)

z= np.array([32.5,29.1]).reshape(1,2)
q=model.predict(z,batch_size=1)
print("첫번째 : 945.75",q)

z= np.array([30,2]).reshape(1,2)
q=model.predict(z,batch_size=1)
print("두번째 : 60",q)

z= np.array([20,10]).reshape(1,2)
q=model.predict(z,batch_size=1)
print("세번째 : 200",q)

#나눗셈
#--------------------------------------------------------------------------------------------------------------------


data_size = 100000 #학습데이터 수
# 학습 범위를 0 ~ 200 까지 발생 시킴 //값을 키울수록 똑똑해지지만 오래걸림
#train_data = np.random.randint(100,size=(data_size,2)) # 0~100 까지를 2 개 더한다.
train_data = np.random.randint(100,size=(data_size,2)) + np.random.rand(data_size, 2)
#-50 ~ +50 범위 출력값이 - 가 나올 수 도 있는상황

train_ans_div=(train_data[:,0]/train_data[:,1])

print("나눗셈",np.shape(train_ans_div))

# 히든 레이어 3개 // 출력은 1개 여야함
# 뺼셈은 어떤 함수를 써야 하는지 고민해봐야함
#모델을 몇개쓸지는 본인 마음이지만 1개로 하는건 상당히 어려움 즉, 그냥 여러개 쓰자
model = keras.Sequential() # 학습데이터의 형태 및 활성화함수의 종류를 고민해봐야함
model.add(keras.layers.Dense(64,activation='tanh'))#중간에 시그모이드는 상관없음 근데 학습이 좀 느림 결과도 부정확함
model.add(keras.layers.Dense(32,activation='tanh')) #레이어는 (1~2)개만 해도 충분한 갯수는 크게 상관없음
# model.add(keras.layers.Dense(5,activation='relu')) #relu는 "-" 표현 안됨 -일때는 고민해 봐야함
model.add(keras.layers.Dense(1,activation='elu')) #시그모이드는  0~1이기에 쓸수없음
#mse? 5+5 예측12 정답은 10 mse --> 4 예측 - 정답 ^ 2 = mse
model.compile(loss='mse',optimizer=tf.optimizers.Adam(0.00029), metrics=['accuracy'])

model.fit(train_data, train_ans_div,batch_size=100,epochs=10)

#문제 10.1 + 20.3 해봐라 소수점은 가르치지 않았음 따라서 소수점은 빼고 계산해서 똑같은 결과가 나옴

z = np.array([29.1,32.5]).reshape(1,2)
q = model.predict(z,batch_size=1)
print("첫번째:0.8953846153846154",q)

z = np.array([30.2,2]).reshape(1,2)
q = model.predict(z,batch_size=1)
print("두번째 : 15.1",q)

z = np.array([20,10]).reshape(1,2)
q = model.predict(z,batch_size=1)
print("세번째:2",q)





















#덧셈 계산기

"""
data_size = 1000 #학습데이터 수
# 학습 범위를 0 ~ 200 까지 발생 시킴 //값을 키울수록 똑똑해지지만 오래걸림
#train_data = np.random.randin(100,size=(data_size,2)) # 0~100 까지를 2 개 더한다.
train_data = np.random.randint(100,size=(data_size,2)) #-50 ~ +50 범위 출력값이 - 가 나올 수 도 있는상황
train_ans=(train_data[:,0]+train_data[:,1])
print(np.shape(train_ans))

# 히든 레이어 3개 // 출력은 1개 여야함
# 뺼셈은 어떤 함수를 써야 하는지 고민해봐야함
#모델을 몇개쓸지는 본인 마음이지만 1개로 하는건 상당히 어려움 즉, 그냥 여러개 쓰자
model = keras.Sequential() # 학습데이터의 형태 및 활성화함수의 종류를 고민해봐야함
model.add(keras.layers.Dense(10,activation='relu'))#중간에 시그모이드는 상관없음 근데 학습이 좀 느림 결과도 부정확함
model.add(keras.layers.Dense(20,activation='relu')) #레이어는 (1~2)개만 해도 충분한 갯수는 크게 상관없음
model.add(keras.layers.Dense(5,activation='relu')) #relu는 "-" 표현 안됨 -일때는 고민해 봐야함
model.add(keras.layers.Dense(1,activation='elu')) #시그모이드는  0~1이기에 쓸수없음
#mse? 5+5 예측12 정답은 10 mse --> 4 예측 - 정답 ^ 2 = mse
model.compile(loss='mse',optimizer=tf.optimizers.Adam(0.00029), metrics=['accuracy'])
model.fit(train_data, train_ans,batch_size=1,epochs=20)

#문제 10.1 + 20.3 해봐라 소수점은 가르치지 않았음 따라서 소수점은 빼고 계산해서 똑같은 결과가 나옴

z= np.array([29.1,32.5]).reshape(1,2)
q=model.predict(z,batch_size=1)
print("첫번째",q)

z= np.array([30.2,-2]).reshape(1,2)
q=model.predict(z,batch_size=1)
print("두번째",q)

z= np.array([100,200]).reshape(1,2)
q=model.predict(z,batch_size=1)
print("세번째",q)
"""







