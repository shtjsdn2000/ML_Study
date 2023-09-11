

#2023_03_29 과제
#Regression : 학습데이터와 - Target 의 상관관계가 있어야함
#두개의 엑셀 파일을 가지고 회귀모델을 하나 만들어볼것
#학습정확도를 얼만큼이상올릴것
#week3에 썼던 리그레이션을 활용할것 (3차정도 무난 // 4,5까지도 볼 수 있음)
#데이터 : 연료가격과 전기가격과의 회귀모델
#대한민국 : 자연에너지 10% / 1. 석탄화력발전소가 제일 많이 사용됨 / 2.원자력 ...석탄과 비슷 비슷/
#/3.석유 /4.LPG 우리나라는 마지막에 사용되는 에너지원의 가격을 따라감 여기선 lpg
#주어진 데이터 종류 : 원자력/ 석유 / lpg
#159행 3열
#아웃풋(타겟) 전기 가격
#0.데이터를받음
#1.데이터를 그려봄 (plot를 활용)
#1-1 그래프를 보고 전기가격과 상관있는지를 판단
#2.3가자의 데이터를 모두 사용해야 될까? 고민해야됨(원자력같은경우는 영향을 거의안줌)
#3.데이터수가 많지않음 따라서 값을 조점하면서 다양한시도를 해봐야함


##벡터연산을 해야하니깐 좀 배워보자
import matplotlib.pyplot as plt
#랜덤 넘버를 만들어 구처럼 만들어보자.
import numpy as np
import matplotlib.pyplot as plt
data_size = 10000 #데이터 값이 커질수록 노이즈가 늘어남
data = np.random.randn(data_size)+1j*np.random.randn(data_size)

#실수와 허수를 랜덤으로 만듦
print(data)
#실수를 x 허수를 y
plt.scatter(data.real, data.imag)
plt.show()
# Q) 1 사분면의 점의 갯수를 구해보자
# 실수 X 가 0보다 작아야함
#데이터 모양을 봄 항상 첫번째! np.shape()는 필수!!!!
print(np.shape(data.real))

#A)정답!
print(np.sum(data.real>0)*(data.imag>0))

# Q) 반지름이 3이 넘어서 있는 점의 갯수를 찾아보자
#np.abs를 사용하면 반지름기준으로 찾아줌
#print(np.abs((data.real>3)*(data.imag>3)))
#A)정답 | 절댓값을 구한후 원점으로 부터 3을 넘어가는 것의 크기를 구해라
print("총 갯수",np.sum(np.abs(data>3)))






