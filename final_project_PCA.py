#final project
#원숭이두창 vs 일반 질병
from PIL.Image import Image
from sklearn.model_selection import train_test_split
#week 10

#https://blog.naver.com/kmh03214/221745095018
#데이터 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from PIL import Image


#케라스 합성곱 층
# from tensorflow import keras
# keras.layers.Conv2D(10,kernel_size=(3,3), activation='relu')
#
# #케라스 패딩 설정
# keras.layers.Conv2D(10, kernel_size=(3,3), activation='relu',padding='same')
#
# #스트라이드
# keras.layers.Conv2D(10, kernel_size=(3,3), activation='relu',padding='same',strides=1)
#
# #폴링
# keras.layers.MaxPooling2D(2)
# keras.layers.MaxPooling2D(2,strides=2, padding='valid')

#CNN을 활용한 데이터처리
#패션 MNIST 데이터
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

#os = director, path 관리 / shutil // 파일을 source->destination 경로로복사
#필요한 라이브러리
import os
import shutil

#이미지를 가져올 경로 설정
# base_dir = '/Users/admin/Downloads' 모르겠음
#이미지 데이터 ->npy로 변형
#https://thinking-developer.tistory.com/62

#변환할 이미지 목록 불러오기
image_path = ('C:\Train_Data\Monkey_VS_other')

img_list = os.listdir(image_path) #디렉토리 내 모든 파일 불러오기
#지정된 확정자만 필터링
img_list_jpg = [img for img in img_list if img.endswith(".jpg")]

print("img_list_jpg:{}".format(img_list_jpg))

img_list_np = []

for i in img_list_jpg:
    img = Image.open(os.path.join(image_path , i))
    img_array = np.array(img)
    img_list_np.append(img_array)
    print(i, " 추가 완료 - 구조:", img_array.shape)  # 불러온 이미지의 차원 확인 (세로X가로X색)
    # print(img_array.T.shape) #축변경 (색X가로X세로)
    #img_np로 저장된듯?
    img_np = np.array(img_list_np)  # 리스트를 numpy로 변환
    print(img_np.shape)# (2142,224,224,3)
#PCA 버전 (비지도 학습)








"""
#https://rfriend.tistory.com/431
def resize_img(img,size):
    return img.resize(size)

def load_img(file_path):
    data =[]
    print(p + file_path[1:] + '/')
    for f in os.listdir(file_path):
        data.append(resize_img(Image.open(p + file_path[1:] + '/' + f ), (64,64)))
        return data

train_M = load_img('C:/Image_data/Test/Monkeypox')
test_M = load_img('C:/Image_data/Train/Monkeypox')
val_M = load_img('C:/Image_data/Val/Monkeypox')

train_NM = load_img('C:/Image_data/Test/Others')
test_NM = load_img('C:/Image_data/Train/Others')
val_NM = load_img('C:/Image_data/Val/Others')

len(train_M),len(test_M),len(val_M),len(train_NM),len(test_NM),len(val_NM)
"""




#합성곱 신경망
"""
#
# (train_input, train_target),(test_input, test_target) = \
#     keras.datasets.fashion_mnist.load_data()
#
# train_scaled = train_input.reshape(-1, 28,28,1) /255.0
#
# train_scaled, val_scaled, train_target, val_target = train_test_split(
#     train_scaled,train_target, test_size=0.2, random_state=42 )
#
# #첫 번째 합성곱 층
# model = keras.Sequential()
#
# model.add(keras.layers.Conv2D(32,kernel_size=3, activation='relu',
#                               padding='same', input_shape=(28,28,1)))
# model.add(keras.layers.MaxPooling2D(2))
#
# #두번째 합성곱 층 + 완전 연결 층
# model.add(keras.layers.Conv2D(64,kernel_size=(3,3), activation='relu', padding='same'))
# model.add(keras.layers.MaxPooling2D(2))
#
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(100, activation ='relu'))
# model.add(keras.layers.Dropout(0,4))
# model.add(keras.layers.Dense(10,activation = 'softmax'))
#
# keras.utils.plot_model(model,show_shapes=True)
#
# #컴파일과 훈련
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics='accuracy')
#
# checkpotint_cb = keras.callbacks.ModelCheckpoint('bast-cnn-model.h5')
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)
#
# history = model.fit(train_scaled, train_target, epochs=20,
#                     validation_data=(val_scaled,val_target),
#                     callbacks=[checkpotint_cb,early_stopping_cb])
#
# #평가와 예측
# model.evaluate(val_scaled,val_target)
#
# plt.imshow(val_scaled[0].reshape(28,28),cmap='gray_r')
# plt.show()
#
# preds = model.predict(val_scaled[0:1])
# print(preds)
#
# #테스트 세트 점수
# test_scaled = test_input.reshape(-1,28,28,1)/255.0
# model.evaluate(test_scaled,test_target)
"""
