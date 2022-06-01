#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os

data_file_path = "E:\Test\demo.csv"

df_data = pd.read_csv(data_file_path) 


# In[2]:


# a few columns have been taken only from the featured dataset for this experiment
select_cols=['Dst Port','Protocol','Flow Duration','Tot Fwd Pkts','Tot Bwd Pkts','TotLen Fwd Pkts','TotLen Bwd Pkts','Fwd Pkt Len Max','Fwd Pkt Len Min','Fwd Pkt Len Std','Bwd Pkt Len Max','Bwd Pkt Len Min','Bwd Pkt Len Std','Flow Byts/s','Flow Pkts/s','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd Header Len','Bwd Header Len','Fwd Pkts/s','Bwd Pkts/s','FIN Flag Cnt','SYN Flag Cnt','RST Flag Cnt','PSH Flag Cnt','ACK Flag Cnt','URG Flag Cnt','Down/Up Ratio','Pkt Size Avg','Fwd Byts/b Avg','Fwd Pkts/b Avg','Bwd Byts/b Avg','Bwd Pkts/b Avg','Init Fwd Win Byts','Init Bwd Win Byts','Active Std','Active Max','Active Min','Label']

select_df_data = df_data[select_cols]

#print(select_df_data)

from sklearn import preprocessing

def prepare_data(df_data):
  df=df_data
  df['Label']=df['Label'].map({'Benign':0, 'DoS attacks-SlowHTTPTest':1}).astype(int)
  ndarray_data = df.values
  features = ndarray_data[:,0:3] # shape varies 
  #print(features)
  label = ndarray_data[:,-1]
  #print(label)

  #minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
  #norm_features = minmax_scale.fit_transform(features)
  #print(norm_features)
  return features, label


shuffled_df_data = select_df_data.sample(frac=1)

x_data, y_data =prepare_data(shuffled_df_data)

#x_data = np.asarry (x_data)
x_data = np.reshape(x_data,x_data.shape + (1,))

#print(x_data)
#print(y_data)

train_size=int(len(x_data)*0.8)

x_train = x_data[:train_size]
y_train = y_data[:train_size]

x_test = x_data[train_size:]
y_test = y_data[train_size:]


# In[3]:


import tensorflow as tf
#tf.__version__

import tensorflow.keras as ks

model = ks.models.Sequential()

model.add(ks.layers.LSTM(units=3, input_shape=(3, 1)))

model.add(ks.layers.Dense(5, activation='relu'))

model.add(ks.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

train_history = model.fit(x_train, y_train, epochs=20, batch_size=12, verbose=1)

score = model.evaluate(x_test, y_test, batch_size=12, verbose=1)

print(score)

# from keras.utils import plot_model
# from keras.layers import Input
# from keras.layers import Dense
# from keras.layers.recurrent import LSTM
# visible = Input(shape=(35,3))
# hidden1 = LSTM(5)(visible)
# hidden2 = Dense(5, activation='relu')(hidden1)
# output = Dense(2, activation='sigmoid')(hidden2)
# model = Model(inputs=visible, outputs=output)



#summarize layers
# print(model.summary())
#plot graph
#plot_model(model, to_file='recurrent_neural_network.png')
#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(units=32, input_dim=5,activation ='relu' ))
#model.add(tf.keras.layers.Dense(units=16, activation ='sigmoid' ))
#model.add(tf.keras.layers.Dense(units=1, activation ='sigmoid' ))


# In[4]:


# model.summary()


# In[5]:

#
# model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss='binary_crossentropy', metrics=['accuracy'])
# train_history = model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=20,batch_size=20,verbose=2 )


# In[6]:


# import matplotlib.pyplot as plt
#
# def visu_train_history(train_history, train_metric, validation_metric):
#   plt.plot(train_history.history[train_metric])
#   plt.plot(train_history.history[validation_metric])
#   plt.title('Train Histroy')
#   plt.ylabel(train_metric)
#   plt.xlabel('epoch')
#   plt.legend(['train', 'validation'], loc='upper left')
#   plt.show()
#
# visu_train_history(train_history, 'acc', 'val_acc')
# visu_train_history(train_history, 'loss', 'val_loss')


# In[ ]:




