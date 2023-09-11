from sklearn.manifold import TSNE
# 판별자에게 진짜와 가짜 이미지를 보여줌
# 판별자 학습을 멈추고 생성자에게 이미지를 만들게함
# 생성자가 만든 이미지를 진짜(target = 1)라고 하며  판별자에게 학습시킴
# 오차가 커진 판별자는 생성자의네트워크(가중치를 변경)를 변경시킴
# 즉 판별자의 영향을 받은 생성자는 점점 진짜같은 이미지를 생성해냄 즉
# 판별자가 판별하기 힘들정도로 진짜같은 이미지가 생성 되어버림

# gan을 활용하는 방안 gan을 활용해 데이터를 만든다.

# 10가지의 패션이미지를 학습시켰는데 신발,가방 이미지가 안나오는 이유
# 나의 생각 1 or 0 으로만 판별자가 판단하기 때문이 아닐까?
# 진짜이유 discriminator가 상의,바지 같은 패턴은 잘 속고 신발,가방은 속지않음
# 따라서 사용하디 어려워짐
# 입력이 random이기 때문에 똑같은걸 겹치게 만들지는 않음
# 예시 가상배우를 gan으로 만듦 이걸로 이제 똑같은 가상배우를 만들어야 하는데 그러지 못함
# 근데 요즘은 기술이 생겨서 가능하다고 함
import numpy as np
import sklearn

assert sklearn.__version__ >= "0.20"
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

np.random.seed(42)
tf.random.set_seed(42)
codings_size = 100  # 생성자와 판별자를 모두 포함하는 MLP 모델을 우선 만든다.

# 디코더를 생성자

# generator = keras.models.Sequential([
#     keras.layers.Dense(100, activation="selu", input_shape=[codings_size]),
#     keras.layers.Dense(150, activation="selu"),
#     keras.layers.Dense(28 * 28, activation="sigmoid"),
#     keras.layers.Reshape([28, 28])
# ])
# decrimminitor ==> output 1 or 0
# discriminator = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(150, activation="selu"),
#     keras.layers.Dense(100, activation="selu"),
#     keras.layers.Dense(1, activation="sigmoid")
# ])

generator = keras.models.Sequential([
    keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
    keras.layers.Reshape([7, 7, 128]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME",
                                 activation="selu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="SAME",
                                 activation="tanh"),
])
discriminator = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME",
                        activation=keras.layers.LeakyReLU(0.2),
                        input_shape=[28, 28, 1]),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME",
                        activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")
])
gan = keras.models.Sequential([generator, discriminator])

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")  # 진짜1 가짜 0 구분하므로 binary_cross사용
discriminator.trainable = False  # 2단계에서 판별자가 가중치 업데이트 안하도록 설정
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")  # GAN도 최종적으로 진짜1 가짜 0 구분하므로 binary_cross사용
# 전체적으로 학습을 하는 것이 아니므로 fit 사용 못함 ➔ 별도의 학습을 위한 함수가 필요함
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)


def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")


def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))  # not shown in the book
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)  # 생성이미지와 실제이미지 합치기
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
        plot_multiple_images(generated_images, 8)  # not shown
        plt.show()


train_gan(gan, dataset, batch_size, codings_size, n_epochs=10)


