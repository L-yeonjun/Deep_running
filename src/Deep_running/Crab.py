#구동에 필요한 케라스 함수
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

#필요한 라이브러리
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

#실행할때 마다 같은 결과를 출력하기 위해 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

# normalization
#데이터 불러오기
df = pd.read_csv('C:/Deep_Learning/dataset/crab.csv', header=None)
#데이터값을 X와 Y로 구분하여 저장
X =df.iloc[:,0:8].values
normalization_df = (X-X.min())/(X.max()-X.min())
X = normalization_df[:,0:8].astype(float)
Y_obj = df.iloc[:, 8].values

# standardization
#데이터 불러오기
df = pd.read_csv('C:/Deep_Learning/dataset/crab.csv', names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
#데이터값을 X와 Y로 구분하여 저장
standardization_df = (df - df.mean())/df.std()
X = standardization_df[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]
Y_obj = df[['i']]


#라벨인코더를 사용해서 Y값의 문자열을 숫자로 변환시킴
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y = tf.keras.utils.to_categorical(Y)

#학습셋과 테스트셋으로 나눔
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = seed)

#모델 설정 (결과값이 3개이상이라서 softmax 사용)
model = Sequential()
model.add(Dense(12, input_dim = 8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='softmax'))

#모델 컴파일 (다중 분류에 적절한 오차함수 인 categorical_crossentropy 사용)
model.compile(loss = 'categorical_crossentropy',
              optimizer ='adam',
              metrics = ['accuracy'])

#모델폴더 생성
MODEL_DIR = 'C:/Users/user5/.spyder-py3/deep_running_crab/crab_acc'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

#저장파일명
modelpath = 'C:/Users/user5/.spyder-py3/deep_running_crab/crab_acc/{epoch:02d} - {accuracy:.4f}.hdf5'

#정확도가 patience 횟수의 epoch동안 더이상 증가하지 않을경우 조기종료
early_stopping_callback = EarlyStopping(monitor = 'accuracy', patience=50)
#값이 더 좋을경우 저장
checkpointer = ModelCheckpoint(filepath = modelpath, monitor = 'accuracy', verbose = 1, save_best_only = True)
#validation_split값만큼 테스트셋 적용
history = model.fit(X_train, Y_train, validation_split=0.3, epochs = 1000, batch_size = 50, verbose = 1, callbacks = [early_stopping_callback, checkpointer])

#X축 값을 0에서 1까지만 표시
plt.ylim([0, 1])

#오차값과 정확도값을 각각 y_vloss와 y_acc에 저장
y_vloss = history.history['val_loss']
y_acc = history.history['accuracy']

#y_vloss(오차값)는 빨간색으로 y_acc(정확도값)는 파랑색으로 그래프에 표시
x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, "-", c = 'red', markersize = 3)
plt.plot(x_len, y_acc, "-", c = 'blue', markersize = 3)

plt.show()

#데스트한 정확도와 오차값을 표시
print('\n Test Accuracy : %.4f' %(model.evaluate(X_test, Y_test)[1]))
print('\n Test loss : %.4f' %(model.evaluate(X_test, Y_test)[0]))