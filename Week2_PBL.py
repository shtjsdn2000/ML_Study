#2023 - 03 - 15
"""
K-최근접 이웃을 이용한 모델을 만들고 보고서에 다음의 내용을 포함하여 제출하시오.

1. 데이터의 클래수 수에 맞게 각 데이터의 군집을 다른색깔로 plt.scatter로 그리시오 (O)

2. predict를 이용하여 임의의 데이터 값이 어떤 클래스로 분류되는지 확인하고,
  분류에 관련된 데이터를 다른색으로 표시하시오. (O)
  이때 K-최근접 이웃 알고리즘에서 고려하는 이웃의 수가 5,10,30,50일때에 대해서 (그림 4개가 나와야 겠지?)
  결과가 차이나는지 그림으로 확인하시오. (0)

3. 본인이 작성한 보고서에 붙여넣기 하세요.

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#입력데이터 불러오기
data_input = np.load('C:/Users/shtjs/data_input.npy')
print(data_input)

print("---------------------------------------------------")
#출력데이터 불러오기
data_target = np.load('C:/Users/shtjs/data_target.npy')
print(data_target)

#데이터를 섞음
np.random.seed(42) #섞는 횟수 42번
index = np.arange(1800)
np.random.shuffle(index)
print(index)

#데이터를 나눠줌
train_input = data_input[index[:1200]]
train_target = data_target[index[:1200]]

test_input = data_input[index[1200:]]
test_target = data_target[index[1200:]]

plt.scatter(train_input[:,0],train_input[:,1])
plt.scatter(test_input[:,0],test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
#총 1800개 데이터가 4개의 클래스로 되어있음 // 학습은 코드를 똑같이 따라가면됨
#중요함 내가 데이터를 적당한 위치에 둬야함 점을 새로 찍을 떄 (22,400)이 적당한 위치
#여기서 주변에 몇개를 볼것이냐에 따라 값이 달라짐 + 넘파이기능을 찾아가면서 넣어봐야됨
#데이터를 500 /400 /400 /500 이런식으로 나눠주면 됨

# fish1_data_size1 = 500
# fish2_data_size1 = 400
# fish3_data_size1 = 400
# fish4_data_size1 = 500


#a = np.sum(data_target==1)+np.sum(data_target==2)+np.sum(data_target==3)+np.sum(data_target==4)
#print(a)

#낯선 생선 추가
distances, indexes = kn.kneighbors([[22,400]])
data_input1 = data_input[0:499]
data_input2 = data_input[500:899]
data_input3 = data_input[900:1299]
data_input4 = data_input[1300:1799]

#scatter로 확인하기

plt.scatter(data_input1[:,0],data_input1[:,1],label ="1")
plt.scatter(data_input2[:,0],data_input2[:,1],label ="2")
plt.scatter(data_input3[:,0],data_input3[:,1],label ="3")
plt.scatter(data_input4[:,0],data_input4[:,1],label ="4")
plt.legend()
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

##------------학습및 데이터 추출하는 코드 시작---------------
#생선을 구별하는 최근접 알고리즘 실행
kn = KNeighborsClassifier(5) #괄호안이 근접갯수 수정 기본값 5
kn = kn.fit(train_input, train_target)
print(kn.score(test_input,test_target))


train_input, test_input, train_target, test_target = train_test_split(
    data_input, data_target, stratify=data_target, random_state=42)


plt.scatter(train_input[:,0],train_input[:,1])
plt.scatter(22,400, marker="^")
plt.scatter(test_input[:,0],test_input[:,1])
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0,800))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 표준점수로 변경
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

print(mean, std)
train_scaled = (train_input - mean) / std

#수상한 생선 다시 표기
new = ([22,400] - mean) / std

#전처리 데이터에서 모델 훈련
kn.fit(train_scaled, train_target)

test_scaled = (test_input - mean) /std
kn.score(test_scaled, test_target)
print("이웃이 5개 일 경우 새로운 생선은 : {}번 입니다." .format(kn.predict([new])))

distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1], marker="^")
plt.scatter(train_scaled[indexes,0],train_scaled[indexes,1],marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
##------------학습및 데이터 추출하는 코드 끝---------------

##------------학습및 데이터 추출하는 코드 시작---------------
#생선을 구별하는 최근접 알고리즘 실행
kn = KNeighborsClassifier(10) #괄호안이 근접갯수 수정 기본값 5
kn = kn.fit(train_input, train_target)
print(kn.score(test_input,test_target))


train_input, test_input, train_target, test_target = train_test_split(
    data_input, data_target, stratify=data_target, random_state=42)

#낯선 생선 추가
distances, indexes = kn.kneighbors([[22,400]])

plt.scatter(train_input[:,0],train_input[:,1])
plt.scatter(22,400, marker="^")
plt.scatter(test_input[:,0],test_input[:,1])
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0,800))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 표준점수로 변경
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

print(mean, std)
train_scaled = (train_input - mean) / std

#수상한 생선 다시 표기
new = ([22,400] - mean) / std

#전처리 데이터에서 모델 훈련
kn.fit(train_scaled, train_target)

test_scaled = (test_input - mean) /std
kn.score(test_scaled, test_target)
print("이웃이 10개 일 경우 새로운 생선은 : {0}번 입니다." .format(kn.predict([new])))

distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1], marker="^")
plt.scatter(train_scaled[indexes,0],train_scaled[indexes,1],marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
##------------학습및 데이터 추출하는 코드 끝---------------

##------------학습및 데이터 추출하는 코드 시작---------------
#생선을 구별하는 최근접 알고리즘 실행
kn = KNeighborsClassifier(30) #괄호안이 근접갯수 수정 기본값 5
kn = kn.fit(train_input, train_target)
print(kn.score(test_input,test_target))


train_input, test_input, train_target, test_target = train_test_split(
    data_input, data_target, stratify=data_target, random_state=42)

#낯선 생선 추가
distances, indexes = kn.kneighbors([[22,400]])

plt.scatter(train_input[:,0],train_input[:,1])
plt.scatter(22,400, marker="^")
plt.scatter(test_input[:,0],test_input[:,1])
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0,800))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 표준점수로 변경
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

print(mean, std)
train_scaled = (train_input - mean) / std

#수상한 생선 다시 표기
new = ([22,400] - mean) / std

#전처리 데이터에서 모델 훈련
kn.fit(train_scaled, train_target)

test_scaled = (test_input - mean) /std
kn.score(test_scaled, test_target)
print("이웃이 30개 일 경우 새로운 생선은 : {}번 입니다." .format(kn.predict([new])))

distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1], marker="^")
plt.scatter(train_scaled[indexes,0],train_scaled[indexes,1],marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
##------------학습및 데이터 추출하는 코드 끝---------------

##------------학습및 데이터 추출하는 코드 시작---------------
#생선을 구별하는 최근접 알고리즘 실행
kn = KNeighborsClassifier(50) #괄호안이 근접갯수 수정 기본값 5
kn = kn.fit(train_input, train_target)
print(kn.score(test_input,test_target))


train_input, test_input, train_target, test_target = train_test_split(
    data_input, data_target, stratify=data_target, random_state=42)

#낯선 생선 추가
distances, indexes = kn.kneighbors([[22,400]])

plt.scatter(train_input[:,0],train_input[:,1])
plt.scatter(22,400, marker="^")
plt.scatter(test_input[:,0],test_input[:,1])
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0,800))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 표준점수로 변경
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

print(mean, std)
train_scaled = (train_input - mean) / std

#수상한 생선 다시 표기
new = ([22,400] - mean) / std

#전처리 데이터에서 모델 훈련
kn.fit(train_scaled, train_target)

test_scaled = (test_input - mean) /std
kn.score(test_scaled, test_target)
print("이웃이 50개 일 경우 새로운 생선은 : {}번 입니다." .format(kn.predict([new])))

distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1], marker="^")
plt.scatter(train_scaled[indexes,0],train_scaled[indexes,1],marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
##------------학습및 데이터 추출하는 코드 끝---------------
