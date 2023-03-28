from rlcompleter import Completer

import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import pywt
Completer.use_jedi = False

scaler=StandardScaler()
x_dataset=[]
y_dataset=[]
src=os.listdir('data_big/notumor')
i=0
for filename in src:
    full_file_name='data_big/notumor/'+filename
    x=cv2.imread(full_file_name)
    b, g, r = cv2.split(x)
    LLb, (LHb, HLb, HHb) = pywt.dwt2(b, 'db2')
    LLg, (LHg, HLg, HHg) = pywt.dwt2(g, 'db2')
    LLr, (LHr, HLr, HHr) = pywt.dwt2(r, 'db2')
    LL = cv2.merge((LLb, LLg, LLr))
    x=np.array(LL,dtype='float32')
    x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    x = cv2.resize(x, (224, 224))
    y=0
    y=np.array(y,dtype='float32')
    x_dataset.append(x)
    y_dataset.append(y)
    i=i+1
    print(i)

src=os.listdir('data_big/glioma')
for filename in src:
    full_file_name='data_big/glioma/'+filename
    x = cv2.imread(full_file_name)
    b, g, r = cv2.split(x)
    LLb, (LHb, HLb, HHb) = pywt.dwt2(b, 'db2')
    LLg, (LHg, HLg, HHg) = pywt.dwt2(g, 'db2')
    LLr, (LHr, HLr, HHr) = pywt.dwt2(r, 'db2')
    LL = cv2.merge((LLb, LLg, LLr))
    x = np.array(LL, dtype='float32')
    x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    x = cv2.resize(x, (224, 224))
    y=1
    y=np.array(y,dtype='float32')
    x_dataset.append(x)
    y_dataset.append(y)
    i=i+1
    print(i)

src=os.listdir('data_big/meningioma')
for filename in src:
    full_file_name='data_big/meningioma/'+filename
    x=cv2.imread(full_file_name)
    b, g, r = cv2.split(x)
    LLb, (LHb, HLb, HHb) = pywt.dwt2(b, 'db2')
    LLg, (LHg, HLg, HHg) = pywt.dwt2(g, 'db2')
    LLr, (LHr, HLr, HHr) = pywt.dwt2(r, 'db2')
    LL = cv2.merge((LLb, LLg, LLr))
    x=np.array(LL,dtype='float32')
    x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    x = cv2.resize(x, (224, 224))
    y=2
    y=np.array(y,dtype='float32')
    x_dataset.append(x)
    y_dataset.append(y)
    i=i+1
    print(i)

src=os.listdir('data_big/pituitary')
for filename in src:
    full_file_name='data_big/pituitary/'+filename
    x=cv2.imread(full_file_name)
    b, g, r = cv2.split(x)
    LLb, (LHb, HLb, HHb) = pywt.dwt2(b, 'db2')
    LLg, (LHg, HLg, HHg) = pywt.dwt2(g, 'db2')
    LLr, (LHr, HLr, HHr) = pywt.dwt2(r, 'db2')
    LL = cv2.merge((LLb, LLg, LLr))
    x=np.array(LL,dtype='float32')
    x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    x = cv2.resize(x, (224, 224))
    y=3
    y=np.array(y,dtype='float32')
    x_dataset.append(x)
    y_dataset.append(y)
    i=i+1
    print(i)

X_dataset=np.array(x_dataset)
Y_dataset=np.array(y_dataset)
Y_dataset=Y_dataset.reshape((7023,1))

from sklearn.model_selection import KFold
folds=list(KFold(n_splits=5,shuffle=True,random_state=1).split(X_dataset,Y_dataset))

Inception=tf.keras.applications.InceptionV3(include_top=False,input_shape=(224,224,3))
input_image=tf.keras.layers.Input((224,224,3))
x=Inception (input_image)
x=tf.keras.layers.Flatten()(x)
hl=tf.keras.layers.Dense(units=4, activation='relu')(x)
x=tf.keras.layers.Dense(4)(hl)
out=tf.keras.layers.Activation(activation='softmax')(x)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model=tf.keras.Model(inputs=input_image,outputs=out)

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

for j, (train_idx, val_idx) in enumerate(folds):
    print("Fold " + str(j + 1))

    x_train = X_dataset[train_idx]
    y_train = Y_dataset[train_idx]
    x_val = X_dataset[val_idx]
    y_val = Y_dataset[val_idx]
    history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[callback], validation_data=(x_val, y_val))
    y_predict = model.predict(x_val)
    Y_predict = np.argmax(y_predict, axis=1)
    l = len(y_val)
    Y_predict = np.reshape(Y_predict, (l, 1))
    prescore = precision_score(y_val, Y_predict, average=None)
    recaller = recall_score(y_val, Y_predict, average=None)
    score = f1_score(y_val, Y_predict, average=None)
    con_mat = confusion_matrix(y_val, Y_predict)
    print("Precision: ", prescore)
    print("Recall: ", recaller)
    print("F1-score ", score)
    print("Confusion Matrix ", con_mat)


