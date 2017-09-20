import numpy as np
import cv2
#import matplotlib.pyplot as plt

def showImage() :

    imgFile = "test.jpg"
    img = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    cv2.namedWindow('model', cv2.WINDOW_NORMAL)
    cv2.imshow('model', img)
    #cv2.waitKey(0)
    k = cv2.waitKey(0) & 0xFF

    if k == 27 :
        cv2.destroyAllWindows()
    elif k == ord('c') :
        cv2.imwrite('test-copy.jpg', img)
        cv2.destroyAllWindows

    #cv2.destroyAllWindows()
showImage()


