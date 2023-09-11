# 다중회귀와 Feature Engineering

#Feature Engineering
#특성과 특성을 조합하여 새로운 특성을 만들어냄
#ex)정면넓이:길이 x 두께 / 옆면넓이: 길이 x 높이

#데이터 입력
import pandas as pd
df = pd.read_csv('https://bit.ly/perch_csv')
perch_full = df.to_numpy()
print(perch_full)

import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

##첫번째 할 일 학습용 데이터와 시험용 데이터 구분
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full,perch_weight,random_state=42)

from sklearn.preprocessing import PolynomialFeatures

#degree=2
poly = PolynomialFeatures()
poly.fit([[2,3]])
#1(bias), 2, 3, 2**2, 2*3, 3**2
print(poly.transform([[2, 3]]))
#[[1.2.3.4.6.9]]
# (degree=5) --> 5차식 까지 써라! 차수가 커질 수록 굴곡이  많아짐 따라서 잠을 정확하게 지나갈 수 있음 정확성향상
# 단점 : 학습데이터에 완벽하게 최적화 되어있기 때문에 테스트 데이터에 낯선 데아터가 들어오면 정확성이 떨어짐 --> 과대적합
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)

poly.get_feature_names_out()

test_poly = poly.transform(test_input)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly,train_target)

print(lr.score(train_poly, train_target))

print(lr.score(test_poly, test_target))

poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)

lr.fit(train_poly,train_target)
print(lr.score(train_poly,train_target))

print(lr.score(test_poly,test_target))
#standarScaler() --> 표준화를 알아서 해줌
##데이터크기 정규화 / 규제
from sklearn.preprocessing import StandardScaler
ss =StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 따라서 등장한것이 Lasso 와 Ridge
# 해결방안 일부러 학습에 오차를 집어넣어줌 따라서
# 그래프가 점을 지나는게 아닌 거의 근접하게 지나게 만듦

##릿지 회귀
#Ridgw(릿지회귀) 결과를 보면 테스트가 상승한걸 알 수 있음
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))

print(ridge.score(test_scaled,test_target))

#그렇다면 오차를 얼마나 넣어야 할까?
#적절한 규제 강도 찾기
import matplotlib.pyplot as plt
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    #릿지 모델을 만듭니다.
    ridge = Ridge(alpha = alpha)
    #릿지 모델을 훈련합니다.
    ridge.fit(train_scaled,train_target)
    #훈련점수와 테스트점수를 지정합니다.
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))


plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled,train_target)

print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))

from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)

print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

print(np.sum(lasso.coef_ == 0))
#리그렛서의 역할 차수의 계수를 알아내는것
# 1.데이터를 주면 몇차식을 쓸건지 고민을 해봐야함
# 2. 차수를 높게 시작하면서 조금씩 낮추면서 적절한 차수를 찾음
#ex)학습정밀도 몇이상
#영향력이 낮은 리그렛서를 없애면서 정밀성을 조금씩 떨굼 = 영향력이 적은걸 없앤다는 뜻
