import sys, cv2
import numpy as np
import copy
from PIL import Image

KEY_ESC = 27

def callback(x):
    pass # do nothing

def usage(progName):
    print('%s blends 2 images with slider.' % progName)
    print('Usgae: >python %s <input image1> <input image2>' % progName)

def main():

    global KEY_ESC

    argv = sys.argv
    argc = len(argv)

    if argc < 3:
        usage(argv[0])
        quit()

    print('Hit ESC-key to terminate this program')

    src1 = Image.open(argv[1]).convert('RGB')
    src1 = np.array(src1)
    src1 = cv2.cvtColor(src1, cv2.COLOR_RGB2BGR)

    src2 = Image.open(argv[2]).convert('RGB')
    src2 = np.array(src2)
    src2 = cv2.cvtColor(src2, cv2.COLOR_RGB2BGR)

    # create a window
    cv2.namedWindow('blend')

    prev_ratio = -1

    # create trackbar
    cv2.createTrackbar('ratio', 'blend', 500, 1000, callback)

    while(1):
        # retrieve the current position of trackbar
        ratio = cv2.getTrackbarPos('ratio', 'blend') 

        if ratio != prev_ratio:

            alpha = (ratio - 500) / 100

            out = cv2.addWeighted(src1, 1.0 - alpha, src2, alpha, 0)
            cv2.imshow('blend', out)

        key = cv2.waitKey(100)
        if(key == KEY_ESC):
            break

    cv2.destroyAllWindows()
    cv2.imwrite('blend.png', out)

if __name__ == '__main__':

    main()
