# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 21:05:09 2021

@author: Emmanuel_Ledezma_H
"""

import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.layers import Input, Flatten, Dense, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import classification_report,confusion_matrix

import os
import cv2

IMG_SIZE = 200
Shapes = ["Circle", "Square", "Triangle", "Star"] 
PATH = ""
Labels = []
Dataset = []


def ObtenerDatosImagenes():
    for shape in Shapes:
        try:
            for path in os.listdir(PATH + shape):
                img = cv2.imread(PATH + shape + '/' + path)
                
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                Dataset.append(img)
                Labels.append(Shapes.index(shape))
        except:
            print(0)
    print("Cantidad dataset:", len(Dataset))



def CrearModelo():
    #Se crean las capas necesarias para el modelo
    input_layer = Input((IMG_SIZE,IMG_SIZE,3))
    x = Flatten()(input_layer)
    
    #x = Dense(200, activation = 'relu')(x)
    x = Dense(64, activation = 'relu')(x)
    
    output_layer = Dense(len(Shapes), activation = 'softmax')(x)
    
    model = Model(input_layer, output_layer)
    model.summary()
    return model



#Metodo que aleatoriamente obtiene 10 imagenes de test y verifica su predicccion con el modelo creado
def PruebaPrediccion():   
    CLASSES = np.array(Shapes)
    preds = Modelo.predict(testX)
    preds_single = CLASSES[np.argmax(preds, axis = -1)]
    actual_single = CLASSES[np.argmax(testY, axis = -1)]
    
    n_to_show = 10
    indices = np.random.choice(range(len(testX)), n_to_show)
    
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    for i, idx in enumerate(indices):
        img = testX[idx]
        ax = fig.add_subplot(1, n_to_show, i+1)
        ax.axis('off')
        ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10, ha='center', transform=ax.transAxes) 
        ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
        ax.imshow(img)    
        

def GraficoAcurracy():
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')


def matrixConfusion():
    CLASSES = np.array(Shapes)
    preds = Modelo.predict(testX)
    preds_single = CLASSES[np.argmax(preds, axis = -1)]
    actual_single = CLASSES[np.argmax(testY, axis = -1)]
    
    matrix = confusion_matrix(preds_single, actual_single)
    print(matrix)


###################### Empieza flujo 
ObtenerDatosImagenes()


# Se normaliza el dataset en imagenes RGB
Dataset = np.array(Dataset)
Dataset = Dataset.astype("float32") / 255.0

Labels = np.array(Labels)
Labels = to_categorical(Labels)

# Se separan las imagenes en conjuntos de test y training (25% train)
(trainX, testX, trainY, testY) = train_test_split(Dataset, Labels, test_size=0.25, random_state=10)

print("X Train :", trainX.shape)
print("X Test :", testX.shape)
print("Y Train :", trainY.shape)
print("Y Test :", testY.shape)

Modelo = CrearModelo()
opt = Adam(lr=0.0001) #El optimizador a usar sera ADAM

#Se compila el modelo a utilizar
Modelo.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Se entrena el modelo mediante la función fit
history = Modelo.fit(trainX
          , trainY
          , batch_size= 90
          , epochs=10
          , shuffle=True)

#Se guarda el modelo creado
Modelo.save("modelTrained.h5");

#matrixConfusion()

#GraficoAcurracy()


#PruebaPrediccion()
    
    
    
    