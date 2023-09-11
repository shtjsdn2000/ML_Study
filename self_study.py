import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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

#도미와 빙어 데이터를 합친다.
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14 #도미 35마리 #빙어 14마리
# 훈련 데이터 입출력 35개
train_input = fish_data[:35]
train_target = fish_target[:35]
# 시험 데이터 입출력 35개
test_input = fish_data[35:]
test_target = fish_target[35:]

#주어진 데이터로 학습 시작
kn = KNeighborsClassifier()
KN = kn.fit(train_input, train_target)

#학습 결과 값 출력
print(kn.score(test_input,test_target))

#훈련데이터를 섞어주기위해기존 1차원 배열(List)을 다차원 배열(Array)로 바꿈
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

input(input_arr)

#데이터를 섞어줌

np.random.seed(42) #왜 42지?
index = np.arrange(49)
np.random.shuffle(index)

np.random.shuffle(index) #내가 볼려고 프린트한거
#index를 랜덤으로 생성하여 도미빙어가 구분되어있는 데이터를 섞어줌
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]


test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]
#데이터를 나누고 확인하기 잘섞여있는지 그래프로 알 수 있음

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(train_input[:,0], train_input[:,1])
plt.xlabel('length')
plt.xlabel('weight')
plt.show()

kn = kn.fit(train_input, train_target)
#훈련세트와 데이터세트의 유사성 나옴 1 에 가까울 수 록 유사함 but 항상 1이 나오진 않음
print(kn.score(test_input, test_target))

#넘파이로 데이터 준비 #Data_Size x [길이,무게] // Data_Size 행 x 2 열
fish_data = np.column_stack((fish_length, fish_weight))

fish_target = np.concatenate(np.ones(35),np.zeros(14))

train_input, test_input, train_target, test_target = train_test_split(
    fish_data,fish_target,stratify=fish_target, random_state=42) # shuffle seed = 42

#사이킷런으로 데이터를 나눈다.
kn.KNeighborsClassifier()
kn.fit(train_input,train_target)
print(kn.score(test_input,test_target))
#특이한 도미를 추가한다.
print(kn.predict([25,100]))

#[25,150]을 판단하는데 사용된 점들의 index와 그들간의 거리 distancs
distances, index = kn.kneighbors([[25,150]])


plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25,150,marker="^")
plt.scatter(train_input[index,0],
train_input[index,1],marker ='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#기준을 맞춰준다.

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25,150,marker="^")
plt.scatter(train_input[index,0],
train_input[index,1],marker ='D')
plt.xlim((0,1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#표준점수로 변경
mean = np.mean(train_input,axis=0)
std = np.std(train_input, axis=0)

print(mean, std)

train_scaled = (train_input - mean) / std
#수상힌 도미 다시 표시하기
new = ([25,150] - mean) / std

plt.scatter(train_scaled[:0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker = "^")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

#전처리 데이터에서 모델 훈련
kn.fit(train_scaled,train_target)
test_scaled = (test_input - mean) / std
kn.score(test_scaled, test_target)

print("새로운 생선" + kn.predict([new]))

distances, indexes = kn.kneighbors(new)
plt.scatter(train_scaled[:0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker = "^")
plt.scatter(train_scaled[indexes,0],train_scaled[indexes,1],marker='D')
plt.xlabel("length")
plt.ylabel("weight")
plt.show()







