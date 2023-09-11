import os

import keras.callbacks
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow_estimator.python.estimator import early_stopping

# 이미지 경로 설정
image_path = ('C:\Train_Data\Monkey_VS_other')

# 이미지 파일 불러오기
img_list = os.listdir(image_path)
img_list_jpg = [img for img in img_list if img.endswith(".jpg")]

# 이미지 데이터와 레이블 저장할 리스트
images = []
labels = []

# 이미지 데이터와 레이블 생성
for img_name in img_list_jpg:
    img = Image.open(os.path.join(image_path, img_name))
    img = img.resize((224, 224))  # 이미지 크기 조정
    img_array = np.array(img)  # 이미지를 배열로 변환
    images.append(img_array)
    #이미지 이름이 M으로 시작하는지 판별
    if img_name.startswith("M"):
        labels.append(1)  # 원숭이 두창 바이러스에 감염된 피부 이미지는 레이블 1
    else:
        labels.append(0)  # 감염되지 않은 피부 병에 감염된 피부 이미지는 레이블 0



# # 데이터셋 분할
train_scaled, val_scaled, train_target, val_target = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# 데이터 전처리
train_images = np.array(train_scaled) / 255.0
test_images = np.array(val_scaled) / 255.0
train_labels = np.array(train_target)
test_labels = np.array(val_target)


# train_image,test_image = input_data / train_labels,test_labels = target_data 

#첫 번째 합성곱 층
model = keras.Sequential()
model.add(keras.layers.Conv2D(32,kernel_size=3,activation = 'relu',
                              padding ='same', input_shape=(224,224,3)))

model.add(keras.layers.MaxPooling2D(2))
#두 번째 합성곱 층 + 완전 연결 층
model.add(keras.layers.Conv2D(64,kernel_size = (3,3), activation='relu',padding ='same'))
model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics='accuracy')

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')

early_stopping_cb = keras.callbacks.EarlyStopping(patience = 2, restore_best_weights=True)

# history = model.fit(train_images, train_labels,epochs=20,
#                    validation_data=(val_scaled,val_target),callbacks=[checkpoint_cb,early_stopping_cb])



# model.evaluate(val_scaled, val_target)

plt.imshow(val_scaled[0].reshape(224,224,3))
plt.show()

preds = model.predict(val_scaled[0:1])
print(preds)