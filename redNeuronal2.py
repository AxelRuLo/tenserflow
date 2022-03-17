import numpy as np
import tensorflow as tf



import pandas as pd 

def leer(archivo:str):
    df = pd.read_csv(archivo,usecols= ['x1','x2','x3','Y'])

    x1 =  np.array(df["x1"].values,"float32")
    x2 =  np.array(df["x2"].values,"float32")
    x3 =  np.array(df["x3"].values,"float32")
    x= []
    for i in range(len(x1)):
        x.append([x1[i],x2[i],x3[i]])

    y =  np.array(df["Y"].values,"float64")

    return x,y


if __name__ == "__main__":
   
    x , y = leer('dataset02.csv')

    #tasa de aprendizaje
    n = 0.1

    model = tf.keras.models.Sequential()
    # pesos aleatorios Adam
    model.add(tf.keras.layers.Dense(1, input_dim=3, activation='linear', kernel_initializer='glorot_uniform', bias_initializer='Ones'))
    
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(n)) 

    historial = model.fit(x, y, epochs=250, verbose=False)
    result = model.predict(x)
    print(result)