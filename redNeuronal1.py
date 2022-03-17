import numpy as np
import tensorflow as tf

import pandas as pd 

def leer(archivo:str):
    df = pd.read_csv(archivo,usecols= ['X','Y'])

    x =  np.array(df["X"].values,"float32")
    y =  np.array(df["Y"].values,"float32")
    # print(x)
    # print(y)
    return x,y


if __name__ == "__main__":
    x , y = leer('dataset01.csv')

# 
    #tasa de aprendizaje
    n = 0.1 
    
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(1, input_dim=1, activation='linear', kernel_initializer='glorot_uniform', bias_initializer='Ones'))
    
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(n)) 
# 
    # Train the perceptron using stochastic gradient descent
    # with a validation split of 20%
    historial = model.fit(x, y, epochs=400, verbose=False)
    result = model.predict(x)
    print(result)

