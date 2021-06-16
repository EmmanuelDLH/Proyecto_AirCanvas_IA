import numpy as np 
import cv2 
from collections import deque 
  
import CanvasIdentification as Recognize

ciclo = 1
Exito, Respuesta = False, "NO SE HA DETECTADO UNA FIGURA VALIDA"


def PrediceImagen(ciclo, frame):
    ciclo = ciclo +1
    if (ciclo % 300 == 0):
        cv2.imwrite('imgCanvas.png',paintWindow)
        imagen = cv2.imread('imgCanvas.png')

        number_of_black_pix = np.sum(imagen == 0) # Se obtiene pixeles negros en imagen
        
        
        if(number_of_black_pix >7000):  #Limite de pixelees para predecir
            gray_image = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('imgCanvas.png',gray_image)
            global Exito, Respuesta
            Exito, Respuesta = Recognize.Predict()
        else:
            Exito, Respuesta = False, "DEBE INCLUIRSE UNA FIGURA DE MAYOR TAMAÑO"

    return ciclo + 1

print()
# default called trackbar function  
def setValues(x): 
   print("") 
   

cv2.namedWindow("Color detectors") 
cv2.createTrackbar("Upper Hue", "Color detectors", 
                   153, 180, setValues) 
cv2.createTrackbar("Upper Saturation", "Color detectors", 
                   255, 255, setValues) 
cv2.createTrackbar("Upper Value", "Color detectors",  
                   255, 255, setValues) 
cv2.createTrackbar("Lower Hue", "Color detectors", 
                   64, 180, setValues) 
cv2.createTrackbar("Lower Saturation", "Color detectors",  
                   72, 255, setValues) 
cv2.createTrackbar("Lower Value", "Color detectors",  
                   49, 255, setValues) 
  
  
# Giving different arrays to handle colour 
# points of different colour These arrays  
# will hold the points of a particular colour 
# in the array which will further be used 
# to draw on canvas 
bpoints = [deque(maxlen = 1024)] 
gpoints = [deque(maxlen = 1024)] 
rpoints = [deque(maxlen = 1024)] 
ypoints = [deque(maxlen = 1024)] 
   
# These indexes will be used to mark position 
# of pointers in colour array 
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
   
# The kernel to be used for dilation purpose  
kernel = np.ones((5, 5), np.uint8) 
  
# The colours which will be used as ink for 
# the drawing purpose 
colors = [(255, 0, 0), (0, 255, 0),  
          (0, 0, 255), (0, 255, 255)] 
colorIndex = 0
   
# Here is code for Canvas setup 
paintWindow = np.zeros((471, 636, 3)) + 255
   
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE) 
   
  
# Loading the default webcam of PC. 
cap = cv2.VideoCapture(0) 

frame = 0   


def DibujaFrame(frame, X, texto):
    y= 30
    for i, line in enumerate(texto.split('\n')):  
        cv2.putText(frame, line, (X,y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 1)
        y = y+20 



# Keep looping 
while True: 
    # Reading the frame from the camera 
    ret, frame = cap.read() 
      
    # Flipping the frame to see same side of yours 
    frame = cv2.flip(frame, 1) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
   
    # Getting the updated positions of the trackbar 
    # and setting the HSV values 
    u_hue = cv2.getTrackbarPos("Upper Hue", 
                               "Color detectors") 
    u_saturation = cv2.getTrackbarPos("Upper Saturation", 
                                      "Color detectors") 
    u_value = cv2.getTrackbarPos("Upper Value", 
                                 "Color detectors") 
    l_hue = cv2.getTrackbarPos("Lower Hue", 
                               "Color detectors") 
    l_saturation = cv2.getTrackbarPos("Lower Saturation", 
                                      "Color detectors") 
    l_value = cv2.getTrackbarPos("Lower Value", 
                                 "Color detectors") 
    Upper_hsv = np.array([u_hue, u_saturation, u_value]) 
    Lower_hsv = np.array([l_hue, l_saturation, l_value]) 
   
   
    # Adding the colour buttons to the live frame  
    # for colour access 
    frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
    cv2.putText(frame, "Clean all", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    

    if(Exito):
        frame = cv2.rectangle(frame, (160,1), (255,65), colors[0], -1)
        frame = cv2.rectangle(frame, (275,1), (370,65), colors[0], -1)
        frame = cv2.rectangle(frame, (390,1), (485,65), colors[0], -1)
        frame = cv2.rectangle(frame, (505,1), (600,65), colors[0], -1)

        DibujaFrame(frame, 180 , Respuesta[0]+"%")
        DibujaFrame(frame, 290 , Respuesta[1]+"%")
        DibujaFrame(frame, 410 , Respuesta[2]+"%")
        DibujaFrame(frame, 520 , Respuesta[3]+"%")

    else:
        cv2.putText(frame, Respuesta, (200, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    

    # Intifying the pointer by making its  
    # mask 
    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv) 
    Mask = cv2.erode(Mask, kernel, iterations = 1) 
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel) 
    Mask = cv2.dilate(Mask, kernel, iterations = 1) 
   
    # Find contours for the pointer after  
    # idetifying it 
    cnts, _ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE) 
    center = None
   
    # Ifthe contours are formed 
    if len(cnts) > 0: 
          
        # sorting the contours to find biggest  
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0] 
          
        # Get the radius of the enclosing circle  
        # around the found contour 
        ((x, y), radius) = cv2.minEnclosingCircle(cnt) 
          
        # Draw the circle around the contour 
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2) 
          
        # Calculating the center of the detected contour 
        M = cv2.moments(cnt) 
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) 
   
        # Now checking if the user wants to click on  
        # any button above the screen  
        if center[1] <= 65: 
              
            # Clear Button 
            if 40 <= center[0] <= 140:  
                bpoints = [deque(maxlen = 512)] 
                gpoints = [deque(maxlen = 512)] 
                rpoints = [deque(maxlen = 512)] 
                ypoints = [deque(maxlen = 512)] 
   
                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0
   
                paintWindow[67:, :, :] = 255

        else : 
            if colorIndex == 0: 
                bpoints[blue_index].appendleft(center) 
            elif colorIndex == 1: 
                gpoints[green_index].appendleft(center) 
            elif colorIndex == 2: 
                rpoints[red_index].appendleft(center) 
            elif colorIndex == 3: 
                ypoints[yellow_index].appendleft(center) 
                  
    # Append the next deques when nothing is  
    # detected to avois messing up 
    else: 
        bpoints.append(deque(maxlen = 512)) 
        blue_index += 1
        gpoints.append(deque(maxlen = 512)) 
        green_index += 1
        rpoints.append(deque(maxlen = 512)) 
        red_index += 1
        ypoints.append(deque(maxlen = 512)) 
        yellow_index += 1
   
    # Draw lines of all the colors on the 
    # canvas and frame  
    points = [bpoints, gpoints, rpoints, ypoints] 
    for i in range(len(points)): 
          
        for j in range(len(points[i])): 
              
            for k in range(1, len(points[i][j])): 
                  
                if points[i][j][k - 1] is None or points[i][j][k] is None: 
                    continue
                      
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2) 
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2) 
   
    cv2.imshow("Tracking", frame) 
    cv2.imshow("Paint", paintWindow) 
    #cv2.imshow("mask", Mask) 
    
    ciclo = PrediceImagen(ciclo,frame)

    # If the 'q' key is pressed then stop the application  
    if cv2.waitKey(1) & 0xFF == ord("q"): 
        bpoints = [deque(maxlen = 512)] 
        gpoints = [deque(maxlen = 512)] 
        rpoints = [deque(maxlen = 512)] 
        ypoints = [deque(maxlen = 512)] 
   
        blue_index = 0
        green_index = 0
        red_index = 0
        yellow_index = 0
   
        paintWindow[67:, :, :] = 255
        
    if cv2.waitKey(1) & 0xFF == ord("e"):
        break
  
# Release the camera and all resources 
cap.release() 
cv2.destroyAllWindows() 