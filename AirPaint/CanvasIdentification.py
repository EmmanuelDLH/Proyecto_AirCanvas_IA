import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


def ProcesarImagen():
    im = cv2.imread("imgCanvas.png")
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    try:
        cnt = contours[1]
        ## this contour is a 3D numpy array\        
        res = cv2.drawContours(im,[cnt],0,(0,0,0), -1)
        cv2.imwrite("contours.png", res)
          # get the 0-indexed coords
        cnt = cnt - cnt.min(axis=0)
        max_xy = cnt.max(axis=0) + 1
        w,h = max_xy[0][0], max_xy[0][1]
        # draw on blank
        canvas = np.ones((h,w,3), np.uint8)*255
        cv2.drawContours(canvas, [cnt], -1, (0,0,0), -1)
        cv2.imwrite("imgCanvas.png", res)
        return True
    except:
        print("No se ha detectado una figura valida")
        return False

    
    
def Predict():  
    
    EsImagenValida = ProcesarImagen()
    tam = 200
    Dataset = []
    
    testData=cv2.resize(cv2.imread("imgCanvas.png"),(tam,tam))
    number_of_black_pix = np.sum(testData == 255)
    print("NUMERO DE PIX ",number_of_black_pix )
    if(number_of_black_pix < 20000 ):
        return False, "NO SE HA DETECTADO UNA FIGURA VALIDA"
    
    Dataset.append(testData)
    Dataset = np.array(Dataset)
    Dataset = Dataset.astype("float32") / 255.0
    
    #Muestra imagen detectada
    plt.imshow(testData,cmap="gray")
    plt.show()
   
                                    
    new = load_model('../Trainig/modelTrained.h5')
    pred=new.predict(Dataset)

    return True, ["Circle:\n"+ TransformaValorPreddiccion(pred[0][0]), 
                  "Square:\n"+ TransformaValorPreddiccion(pred[0][1]), 
                  "Triangle:\n"+ TransformaValorPreddiccion(pred[0][2]),
                  "Star:\n"+ TransformaValorPreddiccion(pred[0][3])]


def TransformaValorPreddiccion(numero):
    entero = numero *100
    return str(float("{:.2f}".format(entero)))








