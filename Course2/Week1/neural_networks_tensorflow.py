from calendar import EPOCH
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

x=np.array([[20,5],[30,5],[15,5],[14,5]])
y=np.array([1,1,1,0])
model = Sequential([Dense(units=25,activation="relu"),Dense(units=15,activation="relu"),Dense(units=1,activation="sigmoid")])
model.compile(loss = tf.keras.losses.binary_crossentropy)
model.fit(x,y,epochs=3000)
a=model.predict(np.array([[20,5],[30,5],[15,5],[14,5]]))
a=model.predict(np.array([[40,5],[12,5],[13,5],[5,5]]))
print(f"{a}")      

