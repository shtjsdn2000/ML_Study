#2023_05_31

import keras.datasets.fashion_mnist

import numpy as np
import sklearn
from keras.layers import activation

assert sklearn.__version__>="0.20"
import tensorflow as tf
import matplotlib.pyplot as plt


import keras.datasets.fashion_mnist

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
X_train = X_train-X_train.mean(axis=0,keepdims=0)

from tensorflow import keras
#활성화 함수는 선형 함수(선형 인코더)를 사용 --> 활성화함수에서 아무것도 안한다는 뜻
encoder = keras.models.Sequential([keras.layers.Dense(2,input_shape=[3])])
decoder = keras.models.Sequential([keras.layers.Dense(3,input_shape=[2])])
autoencoder = keras.models.Sequential([encoder,decoder])
autoencoder.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1.5))
history = autoencoder.fit(X_train,X_train,epochs=20)
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

(X_train_full, y_train_full),(X_test,y_test)=keras.datasets.fashion_mnist.load_data()
#픽셀데이터를 0~1로 바꿔줌
X_train_full = X_train_full.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255
#학습데이터
X_train,X_valid = X_train_full[:-5000],X_train_full[-5000:]
#타겟데이터 but 타겟데이터는 쓸 일이 없음
y_train,y_valid = y_train_full[:-5000],y_train_full[-5000:]

#두 개의 데이터를 넣어 값을 리턴 받음 각각의 값을 넣어 값의 차이를 출력
#y_true = 학습데이터 /

#비교(rounded의 역할)는 어떻게? 픽셀값을 1 or 0 으로 판단 ex)0.5-->1 이라 판단
def rounded_accuracy(y_true,y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true),tf.round(y_pred))

tf.random.set_seed(42)
np.random.seed(42)
#노드생성
#784개를 30개 까지 줄임--> 100까지 늘리고 -->784까지 늘림
stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(100,activation="selu"),
    keras.layers.Dense(30,activation="selu"),
])
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100,activation="selu",input_shape=[30]),
    #마지막이 sigmoid인 이유? --> 애초에 데이터 자체를 0~1로 압축시켰기 때문이다.
    keras.layers.Dense(28*28,activation="sigmoid"),
    keras.layers.Reshape([28,28])
])

stacked_ae = keras.models.Sequential([stacked_encoder,stacked_decoder])
stacked_ae.compile(loss="binary_crossentropy",
                 optimizer=keras.optimizers.SGD(learning_rate=1.5),metrics=[rounded_accuracy])
history = stacked_ae.fit(X_train,X_train,epochs=20,validation_data=(X_valid,X_valid))

def plot_image(image):
    plt.imshow(image,cmap="binary")
    plt.axis("off")
def show_reconstructions(model,images=X_valid,n_images=5):
    reconstructions = model.predict(images[:n_images])
    fig = plt.figure(figsize=(n_images*1.5,3))
    for image_index in range(n_images):
        plt.subplot(2,n_images,1+image_index)
        plot_image(images[image_index])
        plt.subplot(2,n_images, 1+n_images+image_index)
        plot_image(reconstructions[image_index])

show_reconstructions(stacked_ae)
plt.show()
##노드
##784 -> 1000 -> 30 -> 1000 -> 784
## 중간에 1000으로 늘리면 모든 데이터를 손상되는거 없이 전부 저장함...?

##30개의 정보를 2개로 줄임

from sklearn.manifold import TSNE

X_valid_compressed = stacked_encoder.predict(X_valid)
tsne = TSNE()
X_valid_2D = tsne.fit_transform(X_valid_compressed)
X_valid_2D = (X_valid_2D - X_valid_2D.min())/(X_valid_2D.max()-X_valid_2D.min())

plt.scatter(X_valid_2D[:,0],X_valid_2D[:,1],c=y_valid,s=10,cmap="tab10")
plt.axis("off")
plt.show()

#Auto encoder를 위한 선형대수
#layer은 줄일 수록 좋음
#따라서 대칭네트워크인 autoencoder은 중간까지 레이어를 사용하고 나머지는 그대로 복붙한다. 즉 절반까지만 학습해도 가능함
import matplotlib as mpl
plt.figure(figsize=(10,8))
cmap = plt.cm.tab10
plt.scatter(X_valid_2D[:,0],X_valid_2D[:,1],c=y_valid,s=10,cmap=cmap)
image_positions =np.array([[1.,1.]])
for index, position in enumerate(X_valid_2D):
    dist = np.sum((position - image_positions)**2,axis=1)
    if np.min(dist)>0.02: # if far enought from other images
        image_positions = np.r_[image_positions,[position]]
        imagebox = mpl.offsetbox.AnnotationBbox(
            mpl.offsetbox.OffsetImage(X_valid[index], cmap="binary"),
            position,bboxprops={"edgecolor":cmap(y_valid[index]),"lw":2})
        plt.gca().add_artist(imagebox)
        plt.axis("off")
        #save_fig("fashion_mnist_visualization_plot")
        plt.show()

        #AutoEncoder의 가중치 묶기

class DenseTranspose(keras.layers.Layer):
    def _init_(self,dense,activate=None,**kwargs):
        self.dense = dense
        self.activation =keras.activations.get(activation)
        super()._init_(**kwargs)
    def bulid(self,batch_input_shape):
       self.biases = self.add_weight(name="bias",
                                     shape=[self.dense.input_shape[-1]],
                                     initializer="zeros")
       super().build(batch_input_shape)
    def call(self,inputs):
        z = tf.matmul(inputs,self.dense.weight[0],transpose_b=True)
        return self.activation(z+self.biasees)

    #결과를 비교해보자
keras.backend.clear_session()
dense_1 = keras.layers.Dense(100,activation ="selu")
dense_2 = keras.layers.Dense(30,activation ="selu")

tied_encoder = keras.models.Sequential([
    keras.layer.Flatten(input_shape=[28,28]),
    dense_1,
    dense_2
])

tied_decoder = keras.models.Sequential([
    DenseTranspose(dense_2,activation="selu"),
    DenseTranspose(dense_2,activation="selu"),
    keras.layers.Reshape([28,28])
])

tied_ae = keras.models.Sequential([tied_encoder,tied_decoder])

tied_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=1.5),metrics=[rounded_accuracy])
history = tied_ae.fit(X_train,X_train,epochs=10,validation_data=(X_valid,X_valid))

#layer 별 AutoEncoder 학습 방법
#은닉층을 1개만 사용하는 AE를 만드는 함수
def train_autoencoder(n_neurons, X_train,X_valid,loss,optimizer,
                    n_epochs=10,output_activation=None,metrics=None):
    n_inputs = X_train.shape[-1]
    encoder = keras.models.Sequential([
        keras.layers.Dense(n_neurons,activation="selu",input_shape=[n_inputs])
    ])
    decoder = keras.models.Sequential([
        keras.layers.Dense(n_inputs,activation=output_activation),
    ])
    autoencoder = keras.models.Sequential([encoder,decoder])
    autoencoder.compile(optimizer,loss,metrics=metrics)
    autoencoder.fit(X_train,X_train,epochs=n_epochs,
                    validation_data=(X_valid,X_valid))
    return encoder, decoder,encoder(X_train),encoder(X_valid)

#함수를 이용해서 은닉층 하나만 갖는 AE를 2단계까지 적층
K = keras.backend
X_train_flat = K.batch_flatten(X_train) # equivalent to. reshape(-1,28*28)
X_valid_flat = K.batch_flatten(X_valid)
enc1, dec1, X_train_enc1, X_valid_enc1 = train_autoencoder(
    100,X_train_flat,X_valid_flat,"binary_crossentropy",
    keras.optimizers.SGD(learning_rate=1.5),output_activation="sigmoid",
    metrics=[rounded_accuracy])

enc2,dec2,_,_ = train_autoencoder(
    30,X_train_enc1,X_valid_enc1,"mse",keras.optimizers.SGD(learning_rate=0.05),
    output_activation="selu"
)
stacked_ae_1_by_1=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    enc1,enc2,dec2,dec1,
    keras.layers.Reshape([28,28])
])
show_reconstructions(stacked_ae_1_by_1)
plt.show()


