#다양한 회귀 기술들을 적절하게 사용하면서 어떤노력을 했는지를 보는것임
#즉 다양한 회귀 모델을 골고루 사용하면서 적용하면서 이건 왜 잘 풀리고
#이건 왜 잘 안되는지 알아야 할 것 또한 이유까지 왜잘됨? 왜안됨? 이걸 설명해야됨
#얼마나 다양한 회귀모델을 사용했는지가 중점임
#중간고사 날짜 26일부터 오후5시에 시작 시험시간은 아마...1~2시간
#프로젝트발표 19일(수요일)
#기말프로젝트 보건의료창업 ...

from sklearn.model_selection import train_test_split
#여기서 부터 기말고사 범위 및 뉴럴 네트워크

#2023_04_12 week

#인공신경망

# 패션 MNIST
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

(train_input, train_target),(test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

print(train_input.shape, train_target.shape)

print(test_input.shape,test_target.shape)

#입력과 타깃 샘플
#트레인 타겟 속 고유데이터가 몇개인지 볼 수 있음

fig, axs = plt.subplots(1,10, figsize=(10,10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()

print([train_target[i] for i in range(10)])

#의료를 10종으로 분류함
print(np.unique(train_target,return_counts=True))

#로지스틱 회귀
#train_input를 255로 나누는 이유??
#데이터를 0~1사이로 나타내기 위함

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
print(train_scaled.shape)
#sc = SDGClassifier(loss='log', max_iter=5, random_state=42)
#(28*28일렬로 쫙 나열 함 즉 데이터가 행으로 이루어져 있어야함)
#scores = cross_validate(sc, train_scaled, train_target,n_jobs=-1)
#print(np.mean(scores['test_Score']))
#인공신경망 하나의 뉴런이 모든 입력이 연결되어있는구조--> Dense

#케라스 모델 만들기
# 은닉층 갯수가 4개 넘어가면 딥러닝이라 함
from tensorflow import keras
import tensorflow as tf
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target,test_size=0.2, random_state=42)

print(train_scaled.shape, train_target.shape)

print(val_scaled.shape, val_target.shape)

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
model = keras.Sequential(dense)
#에포크 한번  데이처를 한번 쫘악 한번 씩 돌린걸 ==에포크 1번

# 딥러닝
# 이전시간은 레이어를 한개만 씀 따라서 직선을 이용함
# 레이어가 복잡해질 수록 활성화 함수가 반드시 필요함
# 왜? 없으면 한 레이어가 넘어갈 때마다 값이 너무 커져버림 따라서 활성화함수가
#데이터를 살릴지 줄질지 판단해서 크기를 줄여서 보내버림

#심층 신경망

#from tensorflow import keras

(train_input, train_target), (test_input,test_input) = keras.datasets.fashion_mnist.load_data()

from sklearn.model_selection import train_test_split

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

train_scaled, val_scaled, train_target, val_target =train_test_split(
    train_scaled,train_target, test_size=0.2, random_state=42)
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10, activation='softmax')

model = keras.Sequential([dense1, dense2])
















