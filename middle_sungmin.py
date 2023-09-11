import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
fp = pd.read_excel('all_input.xlsx')
ep = pd.read_excel('busan_target.xlsx')

# 주간 요일을 원-핫 인코딩으로 입력 데이터 준비
day_of_week = fp[["요일"]]
one_hot_encoded = pd.get_dummies(day_of_week)
busan_input = fp[["평균기온","최고기온","최저기온","평균습도","최저습도"]]
df = pd.concat([busan_input, one_hot_encoded], axis=1)
busan_target = ep["방문객수"]

# 데이터 정보 출력
print("Input data:\n", df.head(7))
print("Target data:\n", ep.head(7))

#2.불러온 input/target 데이터를 plot를 활용해 비교분석한다.
plt.plot(busan_input)
plt.show()
plt.plot(busan_target)
plt.show()
plt.plot(one_hot_encoded)
plt.show()
plt.plot(ep)
plt.show()
plt.plot(df)
plt.show()

# 훈련 및 테스트 데이터셋 준비
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    df, ep["방문객수"], random_state=42
)

# 데이터 표준화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)
###################################################################

# 선형 회귀 모델 학습 및 평가
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_scaled, train_target)

train_score = lr.score(train_scaled, train_target)
test_score = lr.score(test_scaled, test_target)

print("Linear Regression")
print("Train R^2:", train_score)
print("Test R^2:", test_score)
################################################################

# 다항 회귀 모델 학습 및 평가
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
train_poly = poly.fit_transform(train_input)
test_poly = poly.transform(test_input)

lr = LinearRegression()
lr.fit(train_poly, train_target)

train_score = lr.score(train_poly, train_target)
test_score = lr.score(test_poly, test_target)

print("다항회귀")
print("다항회귀 Train :", train_score)
print("다항회귀 Test :", test_score)

# k-최근접 이웃 회귀 모델 학습 및 평가
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=10)
knr.fit(train_scaled, train_target)

train_score = knr.score(train_scaled, train_target)
test_score = knr.score(test_scaled, test_target)

print("K-최근접 알고리즘")
print("K-최근접 Train :", train_score)
print("K-최근접 Test :", test_score)

# 확률적 경사 하강법 분류기 학습 및 평가
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
sc.fit(train_scaled, train_target)

train_score = sc.score(train_scaled, train_target)
test_score = sc.score(test_scaled, test_target)

print("Stochastic Gradient Descent Classifier")
print("Train accuracy:", train_score)
print("Test accuracy:", test_score)

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#1.데이터를 불러온다.
fp = pd.read_excel('busan_in.xlsx')
ep = pd.read_excel('busan_target.xlsx')
#1-2 넘파이 형태로 변환
busan_input = fp[["평균기온","평균습도"]].to_numpy()
print(busan_input)
busan_target = ep["방문객수"].to_numpy()
print(busan_target)
print("입력되는 변수종류 :",busan_input[:2])

from sklearn.model_selection import train_test_split
train_input, test_input,train_target,test_target = train_test_split(
    busan_input,busan_target,test_size=0.2, random_state=42
    )
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#결정트리
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled,train_target)

print(dt.score(train_scaled,train_target))
print(dt.score(test_scaled, test_target))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()
"""

"""

##############################경계구역###################333
import pandas as pd #pandas 데이터르 불러오느 라이브러리 주로 엑셀 데이터를 불러옴
from sklearn.model_selection import train_test_split #데이터 분리
from sklearn.preprocessing import StandardScaler #데이터를 표준화점수
from sklearn.neighbors import KNeighborsClassifier # 주변에 있는 데이터를 활용 확률표시
import numpy as np
#데이터를 불러옴
fp = pd.read_excel('busan_in.xlsx')
ep = pd.read_excel('all_target.xlsx',dtype={"방문객수":int})
fp.head()
print("pandas:",pd.unique(ep['Species']))
fish_input = fp[['평균기온','평균습도']].to_numpy()
print("입력되는 변수 종류:",fish_input[:2])
fish_target = ep['방문객수'].to_numpy()

#데이터를 분리해줌
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
#데이터 정규화
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
#k근접 알고리즘 사용
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
#데이터의 크기를 보는 것 항상 써야함
#한줄밖에 없는 일차원임
print("데이터의 크기", np.shape(train_target))
#앞에서 10개만 자르겠음
t_target = train_target[0:10]
#t_target Bream이니?
#벡터식을 사용해 for문을 안써도 됨 // 벡터식임
print("방문객 수",np.sum(t_target=='방문객 수'))
###################################################
"""

"""
import pandas as pd

#데이터 준비
#생선에 대한 5개 정보를 --> 7개의 생선종류 데이터로 mapping
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
#5가지 데이터로 알아봄
fish_input = fish[['weight','Length','Diagonal','Height','width']].to_numpy()
fish_target = fish ['Species'].to_numpy()
#k-최근접 이웃의 다중 분류
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled,train_target)

print(kn.classes_)

print(kn.predict(test_scaled[:5]))

proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals = 4))
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

#손실함수
#로지스틱 손실함수
#데이터전처리
import pandas as pd
fish = pd.read_csv('http://bit.ly/fish_csv_data')
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_input,fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss ='log_loss', max_iter=100, random_state=42)
sc.fit(train_scaled,train_target)

print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))

sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

sc = SGDClassifier(loss='log_loss', random_state=42)
train_score = []
test_score = []

classes = np.unique(train_target)
#실핼 횟수를 넉넉히 줌 여기선 300 번 why? 어디서 수렴하는지 모르기 때문
for _ in range(0,300):
    sc.partial_fit(train_scaled, train_target,
                   classes=classes)
    train_score.append(sc.score(train_scaled,
                                train_target))
    train_score.append(sc.score(train_scaled,
                                train_target))
    test_score.append(sc.score(test_scaled,
                               test_target))

    sc = SGDClassifier(loss = 'log_loss', max_iter=100,
                       tol=None, random_state=42)
    sc.fit(train_scaled, train_target)

    print(sc.score(train_scaled,train_target))

    print(sc.score(test_scaled, test_target))
"""


#성민이 코드
"""
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

input_data = pd.read_excel('busan_input.xlsx')
people_data = pd.read_excel('busan_people.xlsx')

input_data.drop(columns=['연번'], inplace=True)
input_data.drop(columns=['평균기온(℃)'],inplace=True)
input_data.columns = ['일시', '최고기온(℃)', '평균습도(%rh)']

people_data.drop(columns=['연번'], inplace=True)
people_data.columns = ['방문일', '방문객수(명)']

people_data.rename(columns={'방문일':'일시'}, inplace=True)

print(people_data.columns)
all = pd.merge(input_data,people_data,how='outer',on='일시')

new_all =  all.set_index('일시')
new_all_graph = new_all[['최고기온(℃)', '평균습도(%rh)', '방문객수(명)']]

fig = plt.figure(figsize=(12,4))
chart = fig.add_subplot(1,1,1)

chart.plot(new_all_graph)
plt.show()

print(all.head())

data = all[['일시', '최고기온(℃)', '평균습도(%rh)','방문객수(명)']].to_numpy()

print(data)

plt.scatter(data[:,0],data[:,1])
plt.scatter(data[:,0],data[:,2])
plt.scatter(data[:,0],data[:,3])
plt.xlabel('시간')
plt.show()

prac1 = all[['최고기온(℃)', '평균습도(%rh)']].to_numpy()
prac2 = all[['방문객수(명)']].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(prac1, prac2, random_state=42)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)

poly.get_feature_names_out()

test_poly = poly.transform(test_input)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly,train_target))

print(lr.score(test_poly, test_target))

poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)

lr.fit(train_poly,train_target)
print(lr.score(train_poly, train_target))

print(lr.score(test_poly,test_target))

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print("학습:",ridge.score(train_scaled, train_target))
print("테스트:",ridge.score(test_scaled, test_target))
print("정확도 차이:",ridge.score(train_scaled, train_target)-ridge.score(test_scaled, test_target))






from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()

knr.fit(train_input,train_target)

print(knr.score(test_input,test_target))

from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)

mae = mean_absolute_error(test_target,test_prediction)
print(mae)

print(knr.score(train_input, train_target))

knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))

"""