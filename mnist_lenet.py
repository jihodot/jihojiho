from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
np.random.seed(3)
weightsPath = "weights/lenet_weights.hdf5"
import lenet
# 1. 데이터셋 준비하기

# 훈련셋과 시험셋 로딩
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 훈련셋과 검증셋 분리
X_val = X_train[50000:]
Y_val = Y_train[50000:]
X_train = X_train[:50000]
Y_train = Y_train[:50000]

X_train = X_train.reshape(50000, 28,28,1).astype('float32') / 255.0
X_val = X_val.reshape(10000, 28,28,1).astype('float32') / 255.0
X_test = X_test.reshape(10000, 28,28,1).astype('float32') / 255.0

# 훈련셋, 검증셋 고르기

# 라벨링 전환
Y_train = np_utils.to_categorical(Y_train)
Y_val = np_utils.to_categorical(Y_val)
Y_test = np_utils.to_categorical(Y_test)
opt = SGD(lr=0.01)
# 2. 모델 구성하기
model = lenet.LeNet.build(width=28, height=28, depth=1, classes=10)

# 3. 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_val, Y_val))