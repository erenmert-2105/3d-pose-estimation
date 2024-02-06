import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow import keras
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns



#%% Read the data

path = 'C:/Users/Hp/Desktop/Bitirme proje/data/'

full_data = pd.DataFrame()

for entry in sorted(os.listdir(path)):
    if os.path.isfile(os.path.join(path, entry)):
        if entry.endswith('.txt'):
            data = pd.read_csv(path+entry,sep=' ',header=None)
            data.drop([129,130],inplace=True,axis=1)
            data['classs'] = entry[-10:-8]
            full_data = pd.concat([full_data,data],ignore_index=True)
            
#%% Split
x = full_data.drop(["classs"],axis=1)
y = full_data.classs.values
x.head()

#%% 12. label is string we need to convert to integger
y = pd.DataFrame(y)
y.iloc[:,0] = y.iloc[:,0].str.replace('t','1')
y.iloc[:,0] = y.iloc[:,0].str.replace('-','2')
y.astype('int32')
#%% train and test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True)
#%% keras dont accept my format so i have to convert it
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#%% Model
early_stop = EarlyStopping(monitor='loss', patience=2)
model = keras.Sequential()

model.add(keras.layers.Dense(128, activation='relu', input_shape=(129,)))


model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dense(32, activation='relu'))

model.add(keras.layers.Dense(16, activation='relu'))

model.add(keras.layers.Dense(64, activation='relu'))

model.add(keras.layers.Dense(32, activation='relu'))

model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dense(64, activation='relu'))

model.add(keras.layers.Dense(16, activation='relu'))


model.add(keras.layers.Dense(13, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




#%% Fitting model
run = model.fit(x_train , y_train , epochs=5,
                 validation_split=0.20, batch_size= 128,callbacks=[early_stop])
model.summary()


from matplotlib import pyplot as plt
print(run.history.keys())

plt.plot(run.history['loss'],label = 'Train loss')
plt.plot(run.history['val_loss'],label = 'Val loss')
plt.legend()
plt.show()



#%% test
#run2 = model.evaluate(x_test,y_test)
#model.summary()
#%% save
#model.save("129+1_3d_version_01")
#%% read model
#model=keras.models.load_model("C:/Users/Hp/Desktop/Bitirme proje/129+1_3d_version_01")

#%% cm with %
#Y_pred = model.predict(x_test)
y_pred=model.predict(x_test) 
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test, y_pred)

f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(cm, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
