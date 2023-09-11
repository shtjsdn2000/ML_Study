#합성곱 신경망
#합성곱 신경망은 주로 이미지 처리에 사용됨
#커널사이즈 #특성맵
#커널사이즈 별로 모아둔 것을 특성맵이라 함

from sklearn.model_selection import train_test_split

#2023_05_09_ week

#인공신경망

# 패션 MNIST
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

(train_input, train_target),(test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28,28,1) /255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled,train_target, test_size=0.2, random_state=42 )

model = keras.Sequential()

model.add(keras.layers.Conv2D(32,kernel_size=3, activation='relu',
                              padding='same', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Conv2D(64,kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation ='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10,activation = 'softmax'))

keras.utils.plot_model(model,show_shapes=True)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics='accuracy')

checkpotint_cb = keras.callbacks.ModelCheckpoint('bast-cnn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20,
                    validation_data=(val_scaled,val_target),
                    callbacks=[checkpotint_cb,early_stopping_cb])

model.evaluate(val_scaled,val_target)

plt.imshow(val_scaled[0].reshape(28,28),cmap='gray_r')
plt.show()

preds = model.predict(val_scaled[0:1])
print(preds)

test_scaled = test_input.reshape(-1,28,28,1)/255.0
model.evaluate(test_scaled,test_target)