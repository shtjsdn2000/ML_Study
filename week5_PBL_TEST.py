# 2023_03_29 과제
# Regression : 학습데이터와 - Target 의 상관관계가 있어야함
# 두개의 엑셀 파일을 가지고 회귀모델을 하나 만들어볼것
# 학습정확도를 얼만큼이상올릴것
# week3에 썼던 리그레이션을 활용할것 (3차정도 무난 // 4,5까지도 볼 수 있음)
# 데이터 : 연료가격과 전기가격과의 회귀모델
# 대한민국 : 자연에너지 10% / 1. 석탄화력발전소가 제일 많이 사용됨 / 2.원자력 ...석탄과 비슷 비슷/
# /3.석유 /4.LPG 우리나라는 마지막에 사용되는 에너지원의 가격을 따라감 여기선 lpg
# 주어진 데이터 종류 : 원자력/ 석유 / lpg
# 159행 3열
# 아웃풋(타겟) 전기 가격
# 0.데이터를받음
# 1.데이터를 그려봄 (plot를 활용)
# 1-1 그래프를 보고 전기가격과 상관있는지를 판단
# 2.3가자의 데이터를 모두 사용해야 될까? 고민해야됨(원자력 같은경우는 영향을 거의안줌)
# 3.데이터수가 많지 않음 따라서 값을 조점하면서 다양한시도를 해봐야함


##############################경계구역###################333
import matplotlib.pyplot as plt
import pandas as pd #pandas 데이터르 불러오느 라이브러리 주로 엑셀 데이터를 불러옴
from sklearn.model_selection import train_test_split #데이터 분리
from sklearn.preprocessing import StandardScaler #데이터를 표준화점수
from sklearn.neighbors import KNeighborsClassifier # 주변에 있는 데이터를 활용 확률표시
import numpy as np
#데이터를 불러옴
#fish = pd.read_csv('https://bit.ly/fish_csv') #.csv 엑셀파일
#fp = pd.read_excel('fp_cut_1.xlsx')
fp2 = pd.read_excel('fp2.xlsx')
ep2 = pd.read_excel('ep2.xlsx')

print(fp2.head())
print(ep2.head())
plt.plot(fp2)
plt.legend(fp2)
plt.show()
plt.plot(ep2)
plt.show()
#영향을 끼치는건 #유류(red) LNG mu
# print("pandas:",pd.unique(fish['Species']))
print("pandas:",pd.unique(ep2['통합']))
# fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
energy_input = fp2[['one','you','mu','ryu','LNG']].to_numpy() #연료단가 / 연량단가 / 연료비
# print("입력되는 물고기 값:",fish_input[:5])
print("입력되는 연료 값:",energy_input[:5])
# fish_target = fish['Species'].to_numpy()
energy_target = ep2['통합']

# train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
train_input, test_input, train_target, test_target = train_test_split(energy_input,energy_target, random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
# kn = KNeighborsClassifier(n_neighbors=3)
# kn.fit(train_scaled, train_target)
# print(kn.score(train_scaled, train_target))
# print(kn.score(test_scaled, test_target))
# #데이터의 크기를 보는 것 항상 써야함
# #한줄밖에 없는 일 차원임
print("데이터의 크기", np.shape(train_target))
# #앞에서 10개만 자르겠음
t_target = train_target[0:10]
# t_target Bream이니?
# #벡터식을 사용해 for문을 안써도 됨 // 벡터식임
print("육지",np.sum(t_target=='육지'))


#k-최근접 이웃의 다중 분류
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# kn = KNeighborsClassifier(n_neighbors=3)
# kn.fit(train_scaled,train_target)
#
# print(kn.classes_)
#
# print(kn.predict(test_scaled[:5]))
#
# proba = kn.predict_proba(test_scaled[:5])
# print(np.round(proba, decimals = 4))

#k-최근접 이웃의 다중분류 주변에 있는 것이 목표가 무엇인지 규정해줌
#1이 나온것은 주변에있는 5개값이 동일하다는 것



# z = a * 무게 + b * 길이 + c * 대각선 + d * 높이 + e * 두께 + f //1차식
# z > 1 을 만들기 위해 abcdef를 조정해 1 을 넘기게 해줌 (음수값을 가질 수 있음)
# ex) z = 도미 // abcdef = 다른 물고기 종류
#도미데이터의 특징을 반영해 도미일경우 z 을 1보다 크게만들지만 다른경우는 1보다 작게 되게끔 알아서 조정해줌
#--> 그것이 로지스틱 회귀
#sigmoid(1/1+e^-z) 함수일 경우 z에 무한대 넣을경우 --> 1 // 마이너스 무한대는 --> 0 에 가까워짐


#로지스틱 회귀 (이진분류)
#Bream 과 Smelt의 갯수를 표시해라!
#                           도미만 True로 표시됨 |빙어만 True로 표시됨
bream_smelt_indexes = (train_target =='Bream') | (train_target == 'Smelt')
#34개 의 데이터가 들어있음 119개중에서 bream / smelt만 가져왔기 때문
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
#도미와 빙어에 대한 데이터만 true로 표시됨

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
#fit을 통해 계산되는것? --> abcdef값 -->왜계산이 될까? 값을 넣었으니깐!
lr.fit(train_bream_smelt, target_bream_smelt) #<--이때 abcde 값을 정함
#2개를 판정하는것이면?
# z에 대한 식이 하나만 있으면 됨 if) 도미일 경우 크게 만들어 주고 아닐 경우 낮게 만들어 주면 되니깐


print(lr.predict(train_bream_smelt[:5]))
#입력 무게 길이 대각선 높이 두꼐
#시그모이드에 대한 결과(z값을 계싼 --> 시그모이드에 입력)
print("로지스틱회귀 (이진분류):",lr.predict_proba(train_bream_smelt[:5]))

#로지스틱 회귀 계수확인
print("로지스틱 회귀 계수확인",lr.coef_, lr.intercept_)

#z값을 구해보면
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
#sigmpoid를 통과한결과
from scipy.special import expit
print(expit(decisions))

#로지스틱회귀(다중분류)
#계수 C를 구할 때까지 값을 1000번만 바꿔봐라 // max_iter를 바꿀 때마다 성능이 달라짐
lr = LogisticRegression(C=20,max_iter = 1000)
lr.fit(train_scaled, train_target) #train_scaled(이전과 달리 7가지 종류 전부 포함)
#여기서 fit은 7가지의 물고기를 모두 찾아야하므로 식이 7개가 필요함
print("로지스틱회귀(다중분류)",lr.score(train_scaled, train_target))
print("로지스틱회귀(다중분류)",lr.score(test_scaled,test_target))

proba = lr.predict_proba(test_scaled[:5])
#확률값 출력 // 실제 확률은 아님
print(np.round(proba,decimals=3))
#7개의 식이 나옴 x가 7개 // 7개를 구분하기 위해선 7개의 식이 필요함
print(lr.coef_.shape, lr.intercept_.shape)

#소프트맥스 함수
decisions = lr.decision_function(test_scaled[:5])
print(np.round(decisions, decimals=2))

from scipy.special import softmax
proba = softmax(decisions, axis=1)
print(np.round(proba,decimals=3))

##확률적 경사 하강법(= SDG)
#손실함수??
#convex --> 아래로 볼록 (최소를 알 수 있음 = 미분하면 0이 되는 점이나옴)
#손실함수가 convex 형태이다? 손실을 최소화 할 수 있는 지점이 존재한다.
#하지만 물고기7가지 찾는 다차식함수는 알기 어려움
#따라서 최저점을 한번에 알기 어려움 abcdef를 바꿔가면서 손실값을 봄
#머신러닝을 돌린다는것은 ? 가정이 필요함
#가정 : 손실함수가 convex 일 것 이다! why? --> 만약에 아니라면 최저점을 찾을 수 없기 때문
#한번 전체를 학습을 한번 시키는것 = 1epoch
#따라서 epoch를 늘릴수록 똑똑해진다아
lr = LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)

proba =lr.predict_proba(test_scaled[:5])
print(np.round(proba,decimals=3))

print(lr.coef_.shape, lr.intercept_.shape)
#
# #손실함수
# #로지스틱 손실함수
# #데이터전처리
# import pandas as pd
# fish = pd.read_csv('http://bit.ly/fish_csv_data')
# fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
# fish_target = fish['Species'].to_numpy()
#
# from sklearn.model_selection import train_test_split
# train_input, test_input, train_target, test_target = train_test_split(
#     fish_input,fish_target, random_state=42)
#
# from sklearn.preprocessing import StandardScaler
# ss = StandardScaler()
# ss.fit(train_input)
# train_scaled = ss.transform(train_input)
# test_scaled = ss.transform(test_input)
#
# from sklearn.linear_model import SGDClassifier
#
# sc = SGDClassifier(loss ='log_loss', max_iter=100, random_state=42)
# sc.fit(train_scaled,train_target)
#
# print(sc.score(train_scaled,train_target))
# print(sc.score(test_scaled,test_target))
#
# sc.partial_fit(train_scaled, train_target)
# print(sc.score(train_scaled, train_target))
# print(sc.score(test_scaled, test_target))
#
# sc = SGDClassifier(loss='log_loss', random_state=42)
# train_score = []
# test_score = []
#
# classes = np.unique(train_target)
# #실핼 횟수를 넉넉히 줌 여기선 300 번 why? 어디서 수렴하는지 모르기 때문
# for _ in range(0,300):
#     sc.partial_fit(train_scaled, train_target,
#                    classes=classes)
#     train_score.append(sc.score(train_scaled,
#                                 train_target))
#     train_score.append(sc.score(train_scaled,
#                                 train_target))
#     test_score.append(sc.score(test_scaled,
#                                test_target))
#
#     sc = SGDClassifier(loss = 'log_loss', max_iter=100,
#                        tol=None, random_state=42)
#     sc.fit(train_scaled, train_target)
#
#     print(sc.score(train_scaled,train_target))
#
#     print(sc.score(test_scaled, test_target))
#
