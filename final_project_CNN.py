import os

import keras.callbacks
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

#CNN사용

#이미지 경로 설정
image_path = ('C:\Train_Data\Monkey_VS_other')

#이미지 파일 불러오기
img_list = os.listdir(image_path)
img_list_jpg = [img for img in img_list if img.endswith(".jpg")]

#이미지 데이터와 레이블 저장할 리스트
images = []
labels = []


#이미지 데이터와 레이블 생성
for img_name in img_list_jpg:
    img = Image.open(os.path.join(image_path, img_name))
#    img = img.resize((224, 224))  #이미지 크기 조정
    img_array = np.array(img)  #이미지를 배열로 변환
    images.append(img_array)

#이미지 이름이 M으로 시작하는지 판별
    if img_name.startswith("M"):
        labels.append(1)  #원숭이 두창 바이러스에 감염된 피부 이미지 = 1
    else:
        labels.append(0)  #감염되지 않은 피부 병에 감염된 피부 이미지 = 0

#몽키 980개 + 낫몽키 1148개 = 총 데이터 2128개
print("데이터의 수",len(images),len(labels))

#데이터 전처리 과정
images = np.array(images) / 255.0
labels = np.array(labels)

#데이터셋 분할
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42)

#CNN 모델 정의
#(3,3)--> 필터의 크기
model = tf.keras.Sequential()
#첫번째 합성곱층
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',padding='same', input_shape=(224, 224, 3)))
model.add(keras.layers.MaxPooling2D(2))
#두번째 합성곱 층 + 완전 연결 층
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu',padding='same', input_shape=(224, 224, 3)))
model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(1, activation='sigmoid'))
#이진분류이기에 sigmoid를 사용 / 또한 0~1사이의 확률을 나타내야하기 때문에 1

keras.utils.plot_model(model, show_shapes=True)
plt.show()



#모델 컴파일과 훈련
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy', # 0,1 로 구분하기에 다중으로 분류하는 crossentropy는 적절하지않음
#               metrics='accuracy')

model.compile(optimizer='adam',
              loss='binary_crossentropy', # 0,1로 구분하기에 이진분류가 적절함 sigmoid와 함께 쓰임
              metrics='accuracy')

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)

# history = model.fit(train_images, train_labels,epochs=20,
#                     validation_data=(val_scaled,val_target),callbacks=[checkpoint_cb,early_stopping_cb])
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

#평가와 예측
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy:', test_acc)

plt.imshow(test_images[0].reshape(224,224,3))
plt.show()

preds1 = model.predict(test_images[0:1])
print(preds1)

# test_scaled = test_images.reshape(-1,224,224,3)/255.0
# model.evaluate(test_scaled, test_labels)

#목표 이미지 경로 설정
# predict_path = (r'C:\Users\shtjs\PycharmProjects\ML_study\NM110_01_00.jpg')
# predict_image = os.listdir(predict_path)
#
#
# plt.imshow(predict_image[0].reshap(224,224,3))
# plt.show()
#
# preds2 = model.predict(predict_image)
# print(preds2)
#
# test_scaled = predict_image.reshape(-1,224,224,3)/255.0
# model.evaluate(test_scaled, test_labels)

#-----------11주차 가중치 시각화-------------

conv = model.layers[0]
print(conv.weights[0].shape,conv.weights[1].shape)

conv_weights = conv.weights[0].numpy()
plt.hist(conv_weights.reshape(-1,1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()



