import numpy as np
import tensorflow as tf


import pandas as pd 

def leer(archivo:str):
    df = pd.read_csv(archivo,usecols= ['X','Y'])

    x =  np.array(df["X"].values,"float32")
    y =  np.array(df["Y"].values,"float64")
    # print(x)
    # print(y)
    return x,y


if __name__ == "__main__":

    x,y = leer('dataset03.csv')
 
    #tasa de aprendizaje
    n = 0.0001
    
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(1, input_dim=1, activation='linear', kernel_initializer='glorot_uniform', bias_initializer='random_uniform'))
    
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(n)) 
# 
    # Train the perceptron using stochastic gradient descent
    # with a validation split of 20%
    historial = model.fit(x, y, epochs=1380, verbose=False)
    result = model.predict([40])
    print(result)

