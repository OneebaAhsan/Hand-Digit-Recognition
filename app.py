#from curses.textpad import rectangle
from cgi import test
from msilib.schema import Font
import pygame, sys
from pygame.locals import *
import numpy as np
from tensorflow.keras.models import load_model
import cv2


#defining colors
white = (255, 255, 255)
black = (0,0,0)
green = (32,178,170)

boundaryinc = 5
image_count = 1
PREDICT = True

#for bg picture
imagesave = False

#load model 
model = load_model('D:/Data-Science-Projects/HandDigitRecog/bestmodel.h5')

#labels to predict the number that is drawn
labels = {0:'Zero', 1:'One', 
            2:'Two', 3:'Three', 
            4:'Four', 5:'Five', 
            6:'Six', 7:'Seven', 
            8:'Eight', 9:'Nine'}

windowsizeX = 640
windowsizeY = 480

#initializing pygame
pygame.init()
font = pygame.font.Font('freesansbold.ttf', 18)
displaysurf = pygame.display.set_mode((windowsizeX, windowsizeY))
WHILE_INT = displaysurf.map_rgb(white)
pygame.display.set_caption("Digit Recognizer")

iswriting = False #for drawing

num_xcord = []  
num_ycord = []

#working of the app 
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(displaysurf, white, (xcord, ycord), 4, 0)  #drawing the circle on the mouse position
            num_xcord.append(xcord)
            num_ycord.append(ycord)

        #for iswriting = True
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            num_xcord = sorted(num_xcord)
            num_ycord = sorted(num_ycord)

            rectangle_min_x, rectangle_max_x = max(num_xcord[0]-boundaryinc, 0), min(windowsizeX, num_xcord[-1]+boundaryinc)  #boundary of the rectangle
            rectangle_min_y, rectangle_max_y = max(num_ycord[0]-boundaryinc, 0), min(num_ycord[-1]+boundaryinc, windowsizeX)    #


            num_xcord = []   #clearing the list
            num_ycord = []

            image_arr = np.array(pygame.PixelArray(displaysurf))[rectangle_min_x:rectangle_max_x, rectangle_min_y:rectangle_max_y].T.astype(np.float32) #creating the image array

            #to save the image
            if imagesave:
                cv2.imwrite("image.png")
                image_count += 1

            if PREDICT:
                image = cv2.resize(image_arr, (28,28))   #model is trained on 28x28 images
                image = np.pad(image, (10,10), 'constant', constant_values = 0)  #padding the image to make it square
                image = cv2.resize(image, (28,28))/255

                label = str(labels[np.argmax(model.predict(image.reshape(1,28,28,1)))]) #predicting the label

                text_surface = font.render(label, True, green, white)
                text_rectangle_obj = text_surface.get_rect()    #getting the rectangle object of the text surface
                text_rectangle_obj.left = rectangle_min_x     #centering the text   
                text_rectangle_obj.bottom = rectangle_max_y    # centering the text

                displaysurf.blit(text_surface, text_rectangle_obj)   #blit is used to diaplay the obj

            if event.type == KEYDOWN:
                if event.unicode == 'n':
                    displaysurf.fill(black)

        
        pygame.display.update()




