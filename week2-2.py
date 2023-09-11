import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#학습데이터와 결과데이터를 나눠줌
from sklearn.model_selection import train_test_split
#데이터 생성 -> 데이터 분리(학습용,시험용)->각각 실행 및 비교확인
bream_data_size=35
smelt_data_size=14

#도미 데이터
bream_length = np.random.randn(1,bream_data_size)*5+35
bream_weight = bream_length*20+np.random.randn(1,bream_data_size)*20

#빙어 데이터
smelt_length= np.random.randn(1,smelt_data_size)*2+12
smelt_weight= smelt_length+np.random.randn(1,smelt_data_size)*2

#두 데이터를 한줄로 재배열
length=np.concatenate((bream_length,smelt_length),axis=1)
weight=np.concatenate((bream_weight,smelt_weight),axis=1)

fish_data = np.concatenate((length,weight),axis=0).transpose()#입력
fish_target = np.array([1]*bream_data_size + [0]*smelt_data_size)#출력

##
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

print(input_arr)

np.random.seed(49)

index = np.arange(49)
np.random.shuffle(index)

train_input = input_arr[[index[:35]]]
train_target = target_arr[[index[:35]]]

##
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# kn = kn.fit(train_input,train_target)
# kn.score(test_input, test_target)

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.xlabel('weight')
plt.show()
#결과가 1.0이 안나올 수 이 있음 어떤 경우인가?
#-->1. 학습모델 자체가 잘못된경우
# ex--> (데이터가 랜덤으로 섞는데 그때 빙어가 2마리 일경우? 5게찍으면 3마리가 도미임 )
#방지하는 방법! --> 학습데이터가 어마무시하게 많아지면됨

#정규화 : x축과 y축의 비율을 동일하게 바꿔주는 것..? // 기준을 맞춰주는 것
#각각의 데이터가 표준편차가 다를때 동일하게 맞추어 주는작업을 정규화라고 한다.
#정규화--> 표준점수로 바꾼다 꼭! 정규화 과정을 거친 후 학습을 시킬 것!!