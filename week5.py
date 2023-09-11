#2023.04.04
#트리 알고리즘
#트리 알고리즘? --> if문을 직접 만들어줌


####################################3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
#DataFrame ==> np ==>
# 표준화 평균을 중심으로 동일한 표준편차
wine = pd.read_csv('wine.csv')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(
data, target, test_size=0.2, random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
##################################################
#wine.info()

#####
###### 정확성이 70% 이기에 로지스틱회귀가 적합하지 않음
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))

print(lr.score(train_scaled,train_target))

print(lr.coef_, lr.intercept_)

###결정트리(새로운 방법 도입)
#과대적합이긴 하지만 어쨌든 분류가 되긴함
#3개의 변수 당도/도수/ph
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
##트리의 깊이가 깊을 수록 과대적합 낮을수록 정확도 하락
#과대적합임 하지만 아까보단 수치가 높아짐
#왜 과대적합? 깊이를 안정해 주었기 때문 따라서 if문을 몇개쓸건지 정해주어야함
print(dt.score(train_scaled, train_target))

print(dt.score(test_scaled, test_target))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()

#결정트리분석
#당도가 마이너스? 왜지?
# ss = StandardScaler() <--여기 평준화 하는 과정에서 평균기준 +- 하기 떄문임
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1,filled=True,
          feature_names=['alcohol','sugar','pH'])
plt.show()


# gini (불순도를 뜻함)
#값이 0.5? 분류를 더 해야함 값이 0? 분류 안해도됨

#가지치기
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled,train_target)

print(dt.score(train_scaled, train_target))

print(dt.score(test_scaled,test_target))

plt.figure(figsize=(20,15))
plot_tree(dt, filled = True,
          feature_names=['alcohol','sugar','pH'])
plt.show()

#스케일 조정하지않은 특성 사용하기
#스탠다드 스케일을 안써도됨 왜?
#왜냐하면 박스 하나당 고려하는 변수는 1가지 입력을 3가지지만 한 박스에서 사용하는 조건은 1개
#스탠다드 스케일을 쓰는경우는 길이/무게 2개를 동시에 고려를 해야하기 때문 하지만 와인은 크고작음을 판단하기 때문..?
#안 써도 결과가 똑같음


#교차 검증과 그리드 서치
#차수가 몇차인지 자동적으로 바뀌면서 찾게 해주는게 그리드 서치

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
#Tree Depth Node 수 Gini
wine = pd.read_csv('wine.csv')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

#검증세트
from sklearn.model_selection import train_test_split

#훈련세트 트레인세트 검증 세트를 2:8로 다 쪼개버림
#데이터 양이 충분 할 때만 이렇게 함
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42 )

sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

#교차검증
from sklearn.model_selection import cross_validate
#교차검증함수
scores = cross_validate(dt, train_input, train_target)
print(scores)

import numpy as np

print(np.mean(scores['test_score']))

#분할기를 사용한 교차검증
from sklearn.model_selection import StratifiedKFold

scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
#n_split = 10 // 10등분해라 몇등분이 적절한지 봐야함
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


#그리드 서치
#모델파라미터에서 학습되는 계수를 조정해주는것
#하이퍼 파라미터? 학습을 위해서 사용자가 설정해 주어야 하는 값

#params --> 5번 값이 바뀌면서 돌아감 값을 미리 설정해줌 // 어떤 값으로 돌릴지 미리 설정하는것
param = []

#랜덤서치??
#랜던서치는 언제쓸까???
# if) 적절한 정확도를 알지 못 할 때 랜덤하게 돌려보도 가장 정확도가 높게 나온 값을 찾는다.
#즉! 적절한 지점을 알지 못할 때 사용
# if) 이미 한번 돌렸는데 값이 낫베드 근데 좀더 나은값을 찾고싶다?? --> 그리드서치

#그래도 값이 안좋다? 트리 클래식 파이를 버린다~! 즉 다른방법을 찾아본다.

# 주의사항 과제
# 웬만해서는 오늘 올릴 것
# 프로젝트 제안서 내야함 다음주 까지 4명 이하로 조원선정
# 리그레이션을 사용 할 것 5주차까지 배운 것을 활용
# 답이 나오겠구나 감을 가지고 해야함
#