# This is a sample Python script.

# 머신러닝이란? if문을 사용하지 않고 입력과 출력의 어떠한 패턴과 규칙을 학습(연관점을 찾음 그것의 역할은 tensorflow가 함)시켜 데이터를 구별 할 줄 아는 것을 말함

#week 1 도미 판별 프로그램 # 2023_03_07~08

#회귀 regression --다음단계--> 뉴럴 네트워크
# 회귀분석 --> 변수들 사이에서 나타나는 경향성을 설명하는것을 목적으로함
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt #matplotlib.pyplot를 plt 형태로 객체화를 해라 라는 뜻

#도미 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0,
34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0,
700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

#빙어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

#(두 생선의 갯수의합은 49마리) 즉,49행 2열 형태로 만들어줘야함

#도미 데이터, 빙어 데이터 합치기 (두 배열의 합)
length = bream_length+smelt_length
weight = bream_weight+smelt_weight

#(두 생선의 갯수의합은 49마리) 즉,49행 2열 형태로 만들어줘야함 //리스트 내포
fish_data = [[l, w] for l, w in zip(length, weight)]

#정답 데이터 //결과
fish_target = [1]*35 + [0]*14

#목표 데이터와 가까운 5개 데이터를 판별해 묶여서 어느 데이터와 가까운지 판별
kn = KNeighborsClassifier() #<-- 주변에 몇개의 데이터로 판별 하는지에 따라 정확성이 달라짐

#입력데이터와 출력 데이터의 관계를 만들어라!
kn.fit(fish_data, fish_target)
print(kn.score(fish_data, fish_target))

#새로운 생선 예측
print(kn.predict([[30,600]]))

kn49 = KNeighborsClassifier(n_neighbors=49)

print(kn49.fit(fish_data,fish_target))
print(kn49.score(fish_data,fish_target))

#산점도(scatter plot)
plt.scatter(bream_length,bream_weight)
plt.scatter(smelt_length,smelt_weight)
plt.xlabel('length')
plt.xlabel('weight')
plt.show()

#0.71 나온 이유?