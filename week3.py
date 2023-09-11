## 3월 29일 / 5월 16일 17시 ~ 19시
##
# 3주차 수업 2023_03_21 // 2023_03_22 다중회귀 와 Feature Engineering
# 오늘은 회귀 함수
# 상관 관계 = 정비례로 증가 한다던가
# ex)(키/발크기),(발크기,손크기)
# 상관없는 관계 (키/재산)
# 기존 데이터들과 '일관성' 있게 예측하는 것이 오늘 공부 할 내용
#분류 와 회귀
#분류 = 가장 가까운것이 무엇인가 찾는 것
#회귀 = 학습데이터와 일관되게 값을 예측 // 데이터가 주어지지 않아도 예측이 가능합

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

#%% 1.데이터 생성
data_size=100
perch_length=np.random.randint(80,440,(1,data_size))/10 #(1,100)
#무게는 길이값의 제곱 이차식으로 만들어놨음
#이차항 1 / 일차항 -20 /  상수항 110
perch_weight=perch_length**2-20*perch_length+110+np.random.randn(1,data_size)*50 #(1,100)


perch_length=perch_length.T
perch_weight=perch_weight.T
#2.학습데이터와 테스트데이터로 쪼개줌
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)
#3.열의 갯수를 1개로 만듦 행은 관심없음
train_input  = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

#4. 현재 길이정보를 기준을 근처의 존재하는 데이터를 선택해 평균을 뽑아냄
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)

#5. 예측이 정확하면 1에 근접함 예측이 target 평균 수준이라면 0가 됨

knr.score(test_input,test_target)


test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)


#학습할 때 정확성이 높아야함 why? 일반적으로 학습데이터로 학습하기 때문에
#과대적합 : 학습 데이터 예측 >>> 테스트데이터 예측
#과소적합 : 학습데이터 예측 < 테스트데이터 예측 << 1
#둘다 원하는 일반적인 상황이 아님
knr.score(train_input,train_target)
knr.score(test_input,test_target)

#이웃 갯수 조정
knr.n_neighbors = 3
knr.fit(train_input, train_target)

print(knr.score(train_input, train_target))

print(knr.score(test_input,test_target))

#선형회귀
# if) 학습데이터로 주어지지 않은 범위의 샘플이 들어오면 큰 오차가 발생
#데이터 사이에 관계를 함수 식(=방정식/y )으로 나타내어 들어온 데이터값(예시로 x or y)을 기반으로 결과를 예측해냄

#50cm 농어의 이웃을 구합니다.
distances, indexes = knr.kneighbors([[50]])

#훈련 세트의 산점도를 그립니다.
plt.scatter(train_input,train_target)

#훈련 세트 중에서 이웃샘플만 다시 그립니다.
plt.scatter(train_input[indexes],train_target[indexes],
            marker='D')

#50cm 농어 데이터
plt.scatter(50,1033,marker='^')
plt.show()

lr = LinearRegression()
#선형 회귀 모델 훈련
lr.fit(train_input,train_target)

print(lr.predict([[50]]))

#길이가 17/18이하면 "-"가 나오는 결과가 문제가 된다./
print(lr.coef_, lr.intercept_)

#선형 회귀 결과 확인하기

#훈련세트의 산점도를 그립니다.
plt.scatter(train_input,train_target)
from sklearn.linear_model import LinearRegression
#15에서 50까지의 1차방정식 그래프를 그립니다.

###plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

# 50cm 농어 데이터
plt.scatter(50,1241.8,marker='^')
plt.show()

#다항회귀
#데이터 쪼개
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))
#2차 함수를 사용
#입력이 제곱으로 들어감
train_poly = np.column_stack((train_input **2 ,train_input))
test_poly = np.column_stack((test_input**2,test_input))
#다항회귀의 결과
lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50**2, 50]]))

print(lr.coef_, lr.intercept_)
#무게 = 1.01 x 길이^2 - 21.6 x 길이 +116.05

#구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다.
point = np.arange(15,50)
#훈련세트의 산점도를 그립니다.
plt.scatter(train_input,train_target)
#15~49까지 2차 방정식 그래프를 그립니다.
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
#50cm 농어 데이터
plt.scatter([50],[1574],marker='^')
plt.show()

print(lr.score(train_poly, train_target))

##print(lr.score(test_poly,train_target))

# Q.다항회귀가 계수를 정확하게 찾기 위해선 어떤 부분이 달라져야 할까?
# A.현재 데이터100개 -->10000만개로 늘린다.
# 데이터가 많을것 / ...직선의 변동 값이 낮은것 // 기울기가 적어야 한다는 뜻일까?
#why? 데이터가 많이 모여일 수록 좀더 촘촘해지기 때문


