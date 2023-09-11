"""
지도 학습(대다수)
학습 데이터와 그것에 대응되는 답이 필요 입력(=학습데이터)과 출력(=타겟)을 둘다줌

비지도 학습(드물다)
입력은 있지만 출력이 없음 / 그렇다면? K-means등을 활용하여 데이터를 가공하여 결과를 만듦
"""
# 2023_03_14~15
#학습용 데이터와 테스트 데이터를 구분하는 방법 / 입력 데이터의 정규화
"""
훈련세트: 학습(훈련)을 위한 학습데이터 및 타겟
-> 정확도를 높이기 위한 목적의 데이터 / 테스트 세트와 중복 X

테스트세트: 훈련결과를 확인하기 위한 입력 데이터 및 타겟
->테스트데이터는 학습에 사용되지않음 / 훈련세트에 대한 정확도와 유사하게 나와야함

--> 구분하려는 데이터가 유사한 비율로 섞여있어야함
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#빙어와 도미가 합쳐진 데이터들
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7,
31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5,
34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0,
38.5, 38.5, 39.5, 41.0, 41.0, #여기 까지가 35개
9.8,10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0,
475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0,
575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0,
920.0, 955.0, 925.0, 975.0, 950.0, #여기 까지가 35개
6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14 #도미 35마리 #빙어 14마리

# np.random.seed(42)
# index = np.arrange(49)
np.random.shuffle(fish_data)

train_input = fish_data[:35]
train_target = fish_target[:35]

test_input = fish_data[35:]
test_target = fish_target[35:]
"""
np.random.seed(42)
index = np.arrange(49)
np.random.shuffle(index)
"""

kn = KNeighborsClassifier()
kn = kn.fit(train_input, train_target)

#새로운 생선 25,150을 측정해보자
print(kn.predict([[25,150]]))

print(kn.score(test_input, test_target))


plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.xlabel('weight')
plt.show()