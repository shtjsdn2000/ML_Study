# 5개의 정보 중 4개를 선택하여 Logistic 회귀를 실시하고
# 어떤 조합일 때 가장 우수한 분류 정확도가 나오는지 알기 위해서
# 각각의 경우에 train과 test set의 분류 정확도를 출력하시오.
# activation 함수는 softmax만 사용

# 프로그램 작성 후 .py 파일만 제출하세요.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

fish = pd.read_csv('/Users/shtjs/PycharmProjects/ML_study/midterm.csv')
fish.head()
print(pd.unique(fish['Species'])) #물고기 종류 7개
# fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()

#조건 1. width 빠짐
fish_input = fish[['Weight','Length','Diagonal','Height']].to_numpy()
print(fish_input[:4])
fish_target = fish[['Species']].to_numpy()
train_input,test_input,test_target,train_target = train_test_split(fish_input,fish_target,random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

lr = LogisticRegression()
lr.fit(train_scaled,train_target)
print("train")
print("train",lr.score(train_scaled,train_target))
print("train",lr.score(test_scaled,test_target))
proba = lr.predict_proba(test_scaled[:4])
print(np.round(proba,decimal =3 ))
print(lr.coef_.shape,lr.intercept_.shape)

decision =lr.decision_function(test_scaled[:4])
print(np.round(decision,decimals=2))
proba =softmax(decision,axis=1)
print(np.round(proba,decimals=3))

#-------------------------------------------------------------
# fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
#조건 2. Height 빠짐
fish_input = fish[['Weight','Length','Diagonal','Width']].to_numpy()
print(fish_input[:4])
fish_target = fish[['Species']].to_numpy()
train_input,test_input,test_target,train_target = train_test_split(fish_input,fish_target,random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

lr = LogisticRegression(c = 20 ,max_iter=1000)
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))
proba = lr.predict_proba(test_scaled[:4])
print(np.round(proba,decimal =3 ))
print(lr.coef_.shape,lr.intercept_.shape)

decision =lr.decision_function(test_scaled[:4])
print(np.round(decision,decimals=2))
proba =softmax(decision,axis=1)
print(np.round(proba,decimals=3))
#----------------------------------------------
# fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
#조건 3. Diagonal 빠짐
fish_input = fish[['Weight','Length','Height','Width']].to_numpy()
print(fish_input[:4])
fish_target = fish[['Species']].to_numpy()
train_input,test_input,test_target,train_target = train_test_split(fish_input,fish_target,random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

lr = LogisticRegression(c = 20 ,max_iter=1000)
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))
proba = lr.predict_proba(test_scaled[:4])
print(np.round(proba,decimal =3 ))
print(lr.coef_.shape,lr.intercept_.shape)

decision =lr.decision_function(test_scaled[:4])
print(np.round(decision,decimals=2))
proba =softmax(decision,axis=1)
print(np.round(proba,decimals=3))
#---------------------------------------------------------------
# fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
#조건 4. Length 빠짐
fish_input = fish[['Weight','Diagonal','Height','Width']].to_numpy()
print(fish_input[:4])
fish_target = fish[['Species']].to_numpy()
train_input,test_input,test_target,train_target = train_test_split(fish_input,fish_target,random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

lr = LogisticRegression(c = 20 ,max_iter=1000)
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))
proba = lr.predict_proba(test_scaled[:4])
print(np.round(proba,decimal =3 ))
print(lr.coef_.shape,lr.intercept_.shape)

decision =lr.decision_function(test_scaled[:4])
print(np.round(decision,decimals=2))
proba =softmax(decision,axis=1)
print(np.round(proba,decimals=3))
#------------------------------------------------------
# fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
#조건 5. Weight 빠짐
fish_input = fish[['Length','Diagonal','Height','Width']].to_numpy()
print(fish_input[:4])
fish_target = fish[['Species']].to_numpy()
train_input,test_input,test_target,train_target = train_test_split(fish_input,fish_target,random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

lr = LogisticRegression(c = 20 ,max_iter=1000)
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))
proba = lr.predict_proba(test_scaled[:4])
print(np.round(proba,decimal =3 ))
print(lr.coef_.shape,lr.intercept_.shape)

decision =lr.decision_function(test_scaled[:4])
print(np.round(decision,decimals=2))
proba =softmax(decision,axis=1)
print(np.round(proba,decimals=3))
