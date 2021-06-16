
"""
Created on Sat May 22 09:44:09 2021

@author: Emmanuel_Ledezma_H
"""

# Se importan las librerías necesarias
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf
import cv2
import os
import numpy as np

labels = ['Circle','Star','Square','Triangle','Pentagon']
#,,, 'Hexagon','Heptagon','Octagon', 'Nonagon'

tamanoImagen = 80 #tamaño de imagenes en caso de redimensionar
epochs= 15
cantidadFiguras =5
valorLr = 0.001 


#Metodo Inicial para realziar la carga de las imagpenes y manipularlas como Arrays
def ObtenerDatosArrayImagenes(tipo):
    data = [] 
    for label in labels: 
        path = label + "/"+tipo
        print(path)
        #Se le asigna un identificador numerico a cada clase de figura
        class_num = labels.index(label)  
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #Se realiza conversión a RGB
                #Se le aplica redimesnsión a las imagenes para que sean más fáciles de utilizar
                resized_arr = cv2.resize(img_arr, (tamanoImagen, tamanoImagen)) 
               
                data.append([resized_arr, class_num]) # usar resized_arr para el array redimensionado
            except Exception as e:
                print("Ha ocurrido excepción ", e)
    return np.array(data,dtype="object")


#Metodo para visualizar los datos de manera gráfica 
def VisualizarGraficoDatos(data):
        plot = []
        for i in data:
            claseFigura = i[1]
            plot.append(labels[claseFigura])
        sns.set_style('darkgrid')
        sns.countplot(plot)


def CreacionModelo():
    model = Sequential()
    model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(tamanoImagen,tamanoImagen,3)))
    model.add(MaxPool2D())
    
    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    
   
    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(cantidadFiguras, activation="softmax"))
    model.summary()
    return model


#Actualmente se utilizaría 71% para train y 29% para test (5000 imagenes por figura)
dataTrain = ObtenerDatosArrayImagenes("Train")
dataTest = ObtenerDatosArrayImagenes("Test")

#Normalización de datos y preprocesamiento previo para el training y test a ralziar 
def PreprocesamientoDatos(tamanoImagen): 
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    #Se separan los datos respectivamente en los distintos arrays (etiqueta y caracteristica)
    for feature, label in dataTrain:
      x_train.append(feature)
      y_train.append(label)
    
    for feature, label in dataTest:
      x_test.append(feature)
      y_test.append(label)
    
    # Se normalizan los arrays de acuerdo al RGB 255
    x_train = np.array(x_train) / 255
    x_test = np.array(x_test) / 255
    
    #Se realiza redimension de imagen para mantener mismos estandares
    x_train.reshape(-1, tamanoImagen, tamanoImagen, 1)
    y_train = np.array(y_train)
    
    x_test.reshape(-1, tamanoImagen, tamanoImagen, 1)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = PreprocesamientoDatos(tamanoImagen)

opt = Adam(lr=valorLr)
model = CreacionModelo()
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) ,
              metrics = ['accuracy'])


#Se debe de buscar el epoch indicado
history = model.fit(x_train,y_train,epochs = epochs , validation_data = (x_test, y_test))
model.save("modelTrained.h5");

#y_true =  model.predict(x_test)
#matriz = confusion_matrix(y_true, y_test)
#print(matriz)






def EvaluacionModelo():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    ##Muestra en tablas los accurracy por figuras y el general
    predictions = model.predict_classes(x_test)
    predictions = predictions.reshape(1,-1)[0]
    print(classification_report(y_test, predictions, 
        target_names = ['Circle (Class 0)','Star (Class 1)','Square (Class 2)',
                        'Triangle (Class 3)','Pentagon (Class 4)'
                        ]))




def ObtenerDatosArrayTest():
    data = []
    #Se le asigna un identificador numerico a cada clase de figura
    try:
        img_arr = cv2.imread("prueba.PNG")[..., ::-1]
                #Se le aplica redimesnsión a las imagenes para que sean más fáciles de utilizar
        resized_arr = cv2.resize(img_arr, (80, 80))

                # usar resized_arr para el array redimensionado
        data.append(resized_arr)
    except Exception as e:
                print("Ha ocurrido excepción ", e)
    return data  # np.asarray(data).astype('float32') # 


def PreprocesamientoDatosTest(array): 
    x_test = []
    for feature in array:
      x_test.append(feature)
    
    x_test = np.array(x_test)/255
    
    x_test.reshape(-1, 80, 80, 1)
    return x_test

EvaluacionModelo()

print("++++++++++++++++++++++")
dataTestt = ObtenerDatosArrayTest()
xx_test = PreprocesamientoDatosTest(dataTestt)
pred = model.predict(xx_test)
predictions = pred.reshape(1,-1)[0]
print("pred",pred)
print("pred 2", predictions)
print(str(pred[0][0]))
print(str(pred[0][1]))
print(str(pred[0][2]))
print(str(pred[0][3]))
print(str(pred[0][4]))




#, ,'Hexagon (Class 5)',
 #                   'Heptagon (Class 6)','Octagon (Class 7)','Nonagon (Class 8)'







