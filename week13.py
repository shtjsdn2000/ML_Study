# 2023_05_30_ week

#AuToEncoder
#MLP와 같음 하지만 인풋아웃풋 노드 수를 같게 해야함 중간노드수는 대칭구조가 아니여도 됨
#특징을 추출하기 위한 용도
#AuTo뜻 자기 자신 즉 자기 자신을 만드는 인코더
#차원축소를 하는 역할을 함
#입력을 출력으로 복사하는 방법을 학습
#Encoding 은 노드가 줄어든다.
#Decoder은 노드가 늘어남 정확히는 복원
#마치 압출했다가 압출을 푼거랑 비슷한 맥락
#입출력은 같지만 중간에 정보량이 줄어드는것
#20일 오후 5시 프로젝트 최종 발표
#평가방법 교수 50 % // 나머지 학생 평가
#6월7일 3시간 연달아 // 6월14일 프로젝트 문의있는 사람만 // 6월20일 5시부터 끝까지 최종발표


#프로그래밍 준비
##텐서 플로우를 가져옴 누가 실해해도 가은 결과를 위해 시드값을 고정
import numpy as np
import sklearn
assert sklearn.__version__>="0.20"
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

##0~1사이의 60개의 값에대가 3을 곱함 --> 0~3사이의 값 * pi-> 2/3pi => 1.5pi 4/3원 -->180도 돌고 90도돌고
def generate_3d_data(m,w1=0.1, w2=0.3, noise=0.1):
    angles = np.random.rand(m)*3*np.pi/2-0.5
    data =np.empty((m,3))
    # 묘한 규칙성이 보이는 랜덤값들
    data[:,0] = np.cos(angles) + np.sin(angles)/2+noise*np.random.randn(m)/2
    data[:,1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
    data[:,2] = data[:,0] * w1 + data[:,1]*w2 + noise*np.random.randn(m)
    return data

X_train = generate_3d_data(60)
X_train = X_train-X_train.mean(axis=0, keepdims=0)

from tensorflow import keras
#활성화 함수는 선형 함수(선형 인코더)를 사용 --> 활성화함수에서 아무것도 안한다는 뜻
encoder = keras.models.Sequential([keras.layers.Dense(2,input_shape=[3])])
decoder = keras.models.Sequential([keras.layers.Dense(3,input_shape=[2])])
autoencoder = keras.models.Sequential([encoder,decoder])
autoencoder.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1.5))
history = autoencoder.fit(X_train,X_train,epochs=50)
decoding = autoencoder(X_train[0:3,:])
print(X_train[0:3,:]) #처음 3개 값
print(decoding[0:3,:])

codings = encoder.predict(X_train)
fig = plt.figure(figsize=(4,3))
plt.plot(codings[:,0],codings[:,1],"b.")
plt.xlabel("$z_1$",fontsize=18)
plt.ylabel("$z_2$",fontsize=18,rotation=0)
plt.grid(True)
plt.show()

# 2023_05_31
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
# 픽셀데이터를 0~1로 바꿔줌
X_train_full = X_train_full.astype(np.float32) / 255
X_train_full = X_test.astype(np.float32) / 255
# 학습데이터
X_train, X_valid = X_train_full[:5000], X_train_full[-5000:]
# 타겟데이터 but 타겟데이터는 쓸 일이 없음
y_train, y_valid = y_train_full[:5000], y_train_full[-5000:]


# 두 개의 데이터를 넣어 값을 리턴 받음 각각의 값을 넣어 값의 차이를 출력
# y_true = 학습데이터 /

# 비교(rounded의 역할)는 어떻게? 픽셀값을 1 or 0 으로 판단 ex)0.5-->1 이라 판단
def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


tf.random.set_seed(42)
np.random.seed(42)
# 노드생성
# 784개를 30개 까지 줄임--> 100까지 늘리고 -->784까지 늘림
stack_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),
])
stack_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    # 마지막이 sigmoid인 이유? --> 애초에 데이터 자체를 0~1로 압축시켰기 때문이다.
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

stack_ae = keras.models.Sequential([stack_encoder, stack_decoder])
stack_ae.compile(loss="binary_crossentropy",
                 optimizer=keras.optimizers.SGD(learning_rate=1.5), metrics=[rounded_accuracy])
history = stack_ae.fit(X_train, epochs=20,
                       validation_data=(X_valid, X_valid))


def plt_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")

