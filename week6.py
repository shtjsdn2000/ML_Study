# 비지도 학습 까지 중간 고사 시험 범위
# 7주차 프로젝트 피드백 및 발표
# 비지도 학습?
# 정해진 타겟이 정해지지 않은 것 (= 학습 데이터로 타겟을 만들고 학습을 시킴 / 기존 에서 타겟을 만드는 과정이 추가됨)

import numpy as np
import matplotlib.pyplot as plt
# 과일 데이터 불러오기
fruits = np.load('fruits_300.npy')
print(fruits.shape) #(300,100,100)(갯수 , 가로, 세로)
#샘플확인
#값이 작을 수록 어두운색
print(fruits[0,0, :])

plt.imshow(fruits[0], cmap='gray')
plt.show()
plt.imshow(fruits[0], cmap='gray_r')
plt.show()

#샘플차원 변경하기
#리쉐잎 기존 2차원 배열을 1차원 배열로 나열
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1,100*100)
banana = fruits[200:300].reshape(-1, 100*100)

print(apple.shape)
# 샘플평균의 히스토그램-픽셀값 분석
# axis = 1 //가로로 더해라 왜? 객체별 색을 알기 위해서
#색이 밝으면 높은값이 많이나옴
#히스토그램을 통해 바나나와 바나나 아닌것을 구별 가능
#여기는 타겟을 만들기 위해서 고민하는 과정
#고민 결과 색상만으로 구별하기가 어렵다. 왜? 사과랑 파인애플 색상이 겹치기 때문
plt.hist(np.mean(apple, axis=1),alpha= 0.8)
plt.hist(np.mean(pineapple, axis=1),alpha= 0.8)
plt.hist(np.mean(banana, axis=1),alpha= 0.8)
plt.legend(['apple','pineapple','banana'])
plt.show()

#픽셀 평균의 히스토그램
# 사과 / 바나나 / 파인애플
#결과를 보면 구별할 수 있는 특징이 없음
fig, axs = plt.subplots(1,3,figsize = (20,5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()

#평균이미지 그리기
#axs = 새로
apple_mean = np.mean(apple,axis=0).reshape(100,100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100,100)
banana_mean = np.mean(banana, axis=0).reshape(100,100)

fig, axs = plt.subplots(1,3,figsize=(20,5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()

#평균과 가까운 사진 고르기

abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
print(abs_mean.shape)

apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i,j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
        axs[i,j].axis('off')
plt.show()

# 군집(Cluster)
# k개의 군집으로 구별 하겠다
# 이런방식이 왜 필요한가? 방대한 양을 사람이 다 파악하기 어렵기 때문임 
# k-Means 알고리즘
# 임의로 k 개의 중심점을 구함 -> 그 주변 가장 가까운 샘플을 찾아냄 -> k군집이 형성
# 군집의 평균을 구하고 새로점을 찍고 반복 


#모델 훈련
from sklearn.cluster import KMeans
#클러스터는 3개 까지만 쓰세요
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d) #fruits_2d 펼친것
print(km.labels_)

#타겟완성!
print(np.unique(km.labels_, return_counts=True))
#(array([0,1,2],dtype = int 32), array([91,98,111]))--> 결과를 보니 그다지 정확하지 않은편임

#첫 번째 클러스터
"""
def draw_fruits(arr,ratio):
    n = len(arr)
    rows = int(np.ceil(n/10))
"""    
# 클러스터 중심
# 점이 3개 데이터 300개 900번 반복 --> 데이터수가 많아질 수 록 복잡도가 상당히 증가함

# 최적의 k 찾기(클러스터의 수 정하기)



##### 프로젝트 과제
# 기존에 있는 공공 데이터들을 활용 / 데이터 갯수는 최소 몇백개 / 각 파라미터당 최소 2개
# ex) 설탕값과 콜라값의 관계 /피부암의 7가지 측정방법의 비교
# 5월 11일 컨펌 =>PCA 끝나고 진행
# 프로젝트 발표 날 / 미정
# 중간고사 시험은 26일
# 중간비율은 10 %
# 기말 프로젝트 40 %

#2023_04_11
#7주차 비지도 학습
#차원축소
#구분에 사용하는 데이터의 개수를 줄이는것
#주성분 분석 과정
#ex) 만약에! 한 친구가 맨날 아아를 마신다..? -->정보량이 적음-->뭐할 지 알 수 있음
#ex) 만약에! 한 친구가 맨날 다른걸 마신다..? -->정보량이 많음-->뭐마실지 알 수 없음
#이용 정보량이 많다...? --> 데이터가 넓게 퍼져 있다는걸 의미

#PCA 분석--> 주성분 분석 --> 갯수 파악 후 주성분 분석을 진행
#PCA 분석을 하면 두가지 정보가나옴
#1. PCA 벡터(Vector) = 주성분 // 300개 사진의 정보량 순서로..
#50개 추출
#만약 대표로 5개 를 뽑는다..? V1,V2,V3,V4,V5 -->주성분 5가지
#(바나나,파인애플,사과) = a1*V1 + a2*V2 + a3*V3 + a4*V4 + a5*V5 <---공통된이미지 종류가 3가지라 가능
# ex) 바나나-->a1,a2 // 파인애플 --> a3,a4 사과-->a5
#바나나사진이100개-->100차원?(X) 차원이란 합칠 수 없는 부분을 말함
#100*100이라서 10000
#공통점이 없는걸로 PCA를 뽑으면 안됨 // 공통적인 패턴이있어야 가능
#공통적인 성분을 하나로 합친다! 이것이 PCA 핵심

#재구성
#배합비율을 알려주고 300개를 복원한다.
#정확성을 높이고 싶다?? pca추출을 50장 보다 더늘리면됨
#50개의 사진만으로 300개를 복원함
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)

fruits_inverse = fruits_inverse.reshape(-1, 100, 100)

#설명된 분산
#추출된 300개의 이미지에 대해서 50개의 핵심이미지를 추출했을때 원래 이미지가 얼마나 들어있는가 확인하는것
print(np.sum(pca.explained_variance_ratio_))

plt.plot(pca.explained_variance_ratio_)

#여기서 질문 만약에 다다른 종류의 사진 300개? 거의 직선일것
#PCA는 공통적인 요소가 많을 때 쓸 수 있으며 또한 꼭 이미지 아니여도 됨
#기말때 PCA를 쓰면 가점을 주신다 ---> 차원축소

#

