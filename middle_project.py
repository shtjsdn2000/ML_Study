import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#1.데이터를 불러온다.
fp = pd.read_excel('all_input.xlsx')
ep = pd.read_excel('busan_target.xlsx')
#1-2 넘파이 형태로 변환

busan_input = fp[["평균기온","최고기온","최저기온","평균습도","최저습도","성수기"]]
#busan_input = fp[["평균기온","최고기온","최저기온"]]
#busan_input = fp["성수기"]

day_of_week = fp[["요일"]]
one_hot_encoded = pd.get_dummies(day_of_week)
busan_hap_input = pd.concat([pd.DataFrame(busan_input), one_hot_encoded], axis=1).to_numpy()
#busan_hap_input = one_hot_encoded.to_numpy()
print("원핫 인코딩 확인")
print(one_hot_encoded)

#print(busan_input)
busan_target = ep["방문객수"].to_numpy()
print(busan_hap_input)


#2.불러온 input/target 데이터를 plot를 활용해 비교분석한다.
plt.plot(busan_input)
plt.show()
# plt.plot(one_hot_encoded)
# plt.show()
plt.plot(ep)
plt.show()
plt.plot(busan_hap_input)
plt.show()

#학습데이터 준비 //학습데이터 shape일치
# df = (기존 + 요일)
from sklearn.model_selection import train_test_split
train_input, test_input,train_target,test_target = train_test_split(
    busan_hap_input,busan_target, random_state=42
    )


# 데이터 표준화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

plt.plot(train_scaled)
plt.show()
plt.plot(test_scaled)
plt.show()

#회귀모델 훈련

#방법1. k-최근접 이웃회귀 사용

############################################################################

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
#knr.fit(train_input,train_target)
knr.fit(train_scaled,train_target)

score = knr.score(test_scaled, test_target)
print("회귀모델 훈련 : ", score)

from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_scaled)
mae = mean_absolute_error(test_target, test_prediction)
print("모델사용결과 평균 오차 : ",mae)
####

#결과
# 회귀모델 훈련 :  0.2810824679567868
# 모델사용결과 평균 오차 :  34467.49565217391
# 상당히 부적확함... 탈락

# 과대적합 과소적합 테스트
print("train : ",knr.score(train_scaled,train_target)) # train :  0.5432073552617944
print("test : ",knr.score(test_scaled,test_target)) # test :  0.2810824679567868

#이웃갯수 조정
knr.n_neighbors = 7
knr.fit(train_scaled,train_target)
print("k-최근접이웃 train : ",knr.score(train_scaled,train_target)) # train :  0.5432073552617944
print("k-최근접이웃 test : ",knr.score(test_scaled,test_target)) # test :  0.2810824679567868

##############################################################################

# 선형회귀에 사용되는 함수
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#lr.fit(train_input,train_target)
lr.fit(train_scaled,train_target)
print("선형회귀 train : ",lr.score(train_scaled,train_target)) # train :  0.5432073552617944
print("선형회귀 test : ",lr.score(test_scaled,test_target)) # test :  0.2810824679567868
#########################################################################################
from sklearn.preprocessing import PolynomialFeatures
#다중회귀
poly = PolynomialFeatures(degree= 15 , include_bias=False)
train_poly = poly.fit_transform(train_input)
test_poly = poly.transform(test_input)

# lr = LinearRegression()
lr.fit(train_poly,train_target)
print("다중회귀 train : ",lr.score(train_poly,train_target))
print("다중회귀 test : ",lr.score(test_poly,test_target))
############################################################################3
"""
# #다항회귀
# from sklearn.preprocessing import PolynomialFeatures
# 
# from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# poly.fit(train_input)
# poly = PolynomialFeatures(include_bias=False)
# train_poly = poly.transform(train_input)
# train_poly = poly.transform(test_input)
# lr = LinearRegression()
# lr.fit(train_poly,train_target)
# 
# print(poly.transform([[7,2]]))
# print("다항회귀 train:",lr.score(train_poly,train_target))
# print("다항회귀 test : ",lr.score(test_poly,test_target))
"""
##########################################################################
#릿지회귀
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
# print("릿지 train : " ,ridge.score(train_scaled,train_target))
# print("릿지 test : " ,ridge.score(test_scaled,test_target))
##########################################################################
# 적절한 규제강도 찾기
train_score =[]
test_score = []
alpha_list = [0.001,0.01,0.1,1,10,100]
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled,train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)

print("릿지train",ridge.score(train_scaled,train_target))
print("릿지test",ridge.score(test_scaled, test_target))
#############################################################################
#라쏘회귀
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled,train_target)

# print("라쏘회귀 train:",lasso.score(train_scaled, train_target))
# print("라쏘회귀 test", lasso.score(test_scaled,test_target))

# 적절한 규제강도 찾기
train_score =[]
test_score = []

alpha_list = [0.001,0.01,0.1,1,10,100]
for alpha in alpha_list:
    lasso = Lasso(alpha=alpha)
    lasso.fit(train_scaled,train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))
##########결과출력###############
plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)

print("라쏘회귀 train:",lasso.score(train_scaled, train_target))
print("라쏘회귀 test",lasso.score(test_scaled, test_target))

print(np.sum(lasso.coef_ == 0))

##

# from sklearn.preprocessing import StandardScaler
# ss = StandardScaler()
# ss.fit(train_input)
# train_scaled = ss.transform(train_input)
# test_scaled = ss.transform(test_input)
#
# plt.plot(train_scaled)
# plt.show()
# plt.plot(test_scaled)
# plt.show()

#SGDClassifier
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
sc.fit(train_scaled, train_target)

print("SGD train : ",sc.score(train_scaled, train_target))
print("SGD test : ",sc.score(test_scaled,test_target))

sc.partial_fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled,test_target))

# 로지스틱 회귀 (다중 분류)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled, test_target))

#################################################33



# #조기종료
# sc = SGDClassifier(loss='log', random_state=42)
# train_score = []
# test_score = []
# classes = np.unique(train_target)
# for _ in range (0,300):
#     sc.partial_fit(train_scaled, train_target, classes = classes)
#     train_score.append(sc.score(train_scaled,train_target))
#     test_score.append(sc.score(test_scaled,test_target))
#
# sc = SGDClassifier(loss='log', max_iter =100,
#                    tol =None, random_state=42)
# sc.fit(train_scaled,train_target)
#
# print(sc.score(train_scaled,train_target))
# print(sc.score(test_scaled, test_target))


"""
#다항특성만들기
#변환기 다항특성을 만들기 위한 도구
#fit() : 새로운 특성의 조합을 찾음 //  transform() : 특성 조합이 반영된 데이터 생성
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()
poly.fit([[2,3]]) #입력데이터 두 개를 줄테니 새로운 조합을 찾음

print(poly.transform([[2,3]])) #조합을 반영해서 새로운 데이터를 만듦

#LinearRegression실행
#데이터들의 각각의 제곱 및 곱 등이 반영된 입력 데이터를 만들었음
poly = PolynomialFeatures(include_bias=False)

poly.fit(train_input)
train_poly = poly


#3.불러온 input/ 데이터를 train/test로 나눠준다.
#4.이때 input와 target를 섞어서 나눠줌
#여기서 3,4,과정을 한번에 진행 // 사이킷런 사용
#5.표준화 진행
"""

"""
import matplotlib.pyplot as plt
import pandas as pd #pandas데이터르 불러오느 라이브러리 주로 엑셀 데이터를 불러옴
from sklearn.model_selection import train_test_split #데이터 분리
from sklearn.preprocessing import StandardScaler #데이터를 표준화점수
from sklearn.linear_model import Lasso #Lasso 회귀
from sklearn.model_selection import RandomizedSearchCV #랜덤서치
import numpy as np
import datetime

# 데이터를 불러옴 그리고 가공
input_data = pd.read_excel('busan_in.xlsx')
target_data = pd.read_excel('all_target.xlsx', dtype={"방문객수":int})

# 입력 변수와 목표 변수를 추출
energy_input = input_data[["평균기온","평균습도"]].to_numpy()
energy_target = target_data["방문객수"].to_numpy()

# 데이터 분리
train_input, test_input, train_target, test_target = train_test_split(energy_input, energy_target, random_state=42)

# 데이터 표준화
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()
poly.fit([[2,3]])

print(poly.transform([2,3]))

# Lasso 회귀 모델 생성
lasso = Lasso(max_iter=10000)

# alpha 값의 범위 설정
param_distribution = {
    'alpha': np.logspace(-4, 2, 100)
}

# 랜덤서치
random_search = RandomizedSearchCV(lasso, param_distribution, n_iter=100, cv=5, n_jobs=-1, random_state=42)
random_search.fit(train_scaled, train_target)

# 최적의 alpha 값 출력
print('최적의 alpha 값:', random_search.best_params_['alpha'])

# 최적의 alpha 값을 사용하여 Lasso 회귀 모델 학습
lasso = Lasso(alpha=random_search.best_params_['alpha'], max_iter=10000) # alpha 값 조정 가능
lasso.fit(train_scaled, train_target)

# 학습 정확도, 테스트 정확도 출력
print("학습 정확도 :", lasso.score(train_scaled, train_target))
print("테스트 정확도 :", lasso.score(test_scaled, test_target))
print("정확도 차이 : ", lasso.score(train_scaled, train_target) - lasso.score(test_scaled, test_target))
"""

#ver1

"""
import matplotlib.pyplot as plt
import pandas as pd #pandas데이터르 불러오느 라이브러리 주로 엑셀 데이터를 불러옴
from sklearn.model_selection import train_test_split #데이터 분리
from sklearn.preprocessing import StandardScaler #데이터를 표준화점수
from sklearn.neighbors import KNeighborsClassifier #주변에 있는 데이터를 활용 확률표시
import numpy as np
import datetime
#데이터를 불러옴 그리고 가공
fp = pd.read_excel('busan_in.xlsx')
ep = pd.read_excel('all_target.xlsx',dtype={"방문객수":int})
#fp['일시'] = pd.to_datetime(fp['일시'])

#fp['요일'] = fp['일시'].dt.weekday

print(fp)

ep.head()
plt.plot(ep)
#plt.show()
plt.plot(fp)
plt.show()
plt.plot(ep,fp)
plt.show
print("pandas:",pd.unique(ep['방문객수']))

energy_input = fp[["평균기온","평균습도"]].to_numpy() # #fp = fp.filter(["무연탄","유류","LNG"])
print("입력되는 변수종류 :",energy_input[:2])
# fish_target = fish['Species'].to_numpy()
energy_target = ep["방문객수"].to_numpy()

#데이터분리
train_input, test_input, train_target, test_target = train_test_split(energy_input,energy_target, random_state=42)
#데이터 표준화
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

print("데이터의 크기", np.shape(train_target))
# #앞에서 10개만 자르겠음
t_target = train_target[0:2]
t_target = test_target.astype("int")
# #벡터식을 사용해 for문을 안써도 됨 //벡터식임
#print("방문객수",np.sum(t_target=="방문객수"))
print("방문객수", np.sum(t_target==1))
# z = a *무게 + b *길이 + c *대각선 + d *높이 + e *두께 + f //1차식
# z > 1을 만들기 위해 abcdef를 조정해 1을 넘기게 해줌 (음수값을 가질 수 있음)
# ex) z =도미 // abcdef =다른 물고기 종류
#도미데이터의 특징을 반영해 도미일경우 z을 1보다 크게만들지만 다른경우는 1보다 작게 되게끔 알아서 조정해줌
#-->그것이 로지스틱 회귀
#sigmoid(1/1+e^-z)함수일 경우 z에 무한대 넣을경우 --> 1 //마이너스 무한대는 --> 0에 가까워짐


#로지스틱 회귀 (이진분류)
#Bream과 Smelt의 갯수를 표시해라!
from sklearn.linear_model import LinearRegression
#fit을 통해 계산되는것? --> abcdef값 -->왜계산이 될까?값을 넣었으니깐!
#lr.fit(train_bream_smelt, target_bream_smelt) #<--이때 abcde값을 정함

lr = LinearRegression()
lr.fit(train_scaled, train_target)

#학습 정확도,테스트 정확도 출력
print("학습 정확도 :", lr.score(train_scaled, train_target))
print("테스트 정확도 :", lr.score(test_scaled, test_target))
print("정확도 차이 : ",lr.score(train_scaled, train_target)-lr.score(test_scaled, test_target))
"""