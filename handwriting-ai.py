
import cv2
import numpy as np
import pygame, sys
from pygame import image
from pygame.locals import *
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import constant
from numpy import testing
from tokenize import Number
from numpy.lib.type_check import imag
from tensorflow.python.keras.backend import constant


WINDOWSIZEX = 1000
WINDOWSIZEY = 900


BOUNDRYINC = 5
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)

IMAGESAVE = False

MODEL = load_model("bestmodel.h5")

LABELS = {0:"Sıfır", 1:"Bir", 2:"İki",3:"Üç", 4:"Dört",5:"Beş",6:"Altı",7:"Yedi",8:"Sekiz",9:"Dokuz"}

pygame.init()

FONT = pygame.font.Font("C:\Windows\Fonts\Arial.ttf", 18)


DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption("Dijital Ekran")

iswriting = False

number_xcord = []
number_ycord = []

image_cnt = 1
PREDICT = True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting: 
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord,ycord), 4 , 0)   

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x , rect_max_x, = max(number_xcord[0]- BOUNDRYINC, 0 ), min(WINDOWSIZEX, number_xcord[-1]+ BOUNDRYINC)
            rect_min_y , rect_max_y, = max(number_ycord[0]- BOUNDRYINC, 0 ), min(WINDOWSIZEX, number_ycord[-1]+ BOUNDRYINC)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x , rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28,28))
                image = np.pad(image, (10,10), 'constant', constant_values = 0)
                image= cv2.resize(image, (28,28))/255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left , textRecObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurface, textRecObj)

            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURF.fill(BLACK)

        pygame.display.update()

