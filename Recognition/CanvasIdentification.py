import keras
import cv2
import os
import numpy as np


def ObtenerDatosArrayImagenes():
    data = []
    #Se le asigna un identificador numerico a cada clase de figura
    try:
        img_arr = cv2.imread(os.path.join("../Trainig/prueba.PNG"))[..., ::-1]
                #Se le aplica redimesnsión a las imagenes para que sean más fáciles de utilizar
        resized_arr = cv2.resize(img_arr, (60, 60))

                # usar resized_arr para el array redimensionado
        data.append(resized_arr)
    except Exception as e:
                print("Ha ocurrido excepción ", e)
    return data  # np.asarray(data).astype('float32') # 


def PreprocesamientoDatos(array): 
    x_test = []
    for feature in array:
      x_test.append(feature)
    
    x_test = np.array(x_test)/ 255
    
    x_test.reshape(-1, 60, 60, 1)
    return x_test

def Prueba():

    dataTest = ObtenerDatosArrayImagenes()
    x_test = PreprocesamientoDatos(dataTest)
    model = keras.models.load_model('../Trainig/model.h5')
    pred = model.predict(x_test)
    print("pred",pred)
    print(str(pred[0][0]))
    print(str(pred[0][1]))
    print(str(pred[0][2]))
    print(str(pred[0][3]))
    print(str(pred[0][4]))



Prueba()




