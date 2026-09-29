import cv2, sys
import numpy as np
import matplotlib.pyplot as plt

def createHisto0(img):

    H, W = img.shape[:2]
    
    histogram = np.zeros((3, 256), np.int32)
    for y in range(H):
        for x in range(W):
            b = img[y][x][0] 
            g = img[y][x][1] 
            r = img[y][x][2] 
    
            histogram[0][b] += 1
            histogram[1][g] += 1
            histogram[2][r] += 1
    
    return histogram

def createHisto1(img):

    histogram = []
    for ch in range(3):
        histogram.append(cv2.calcHist([img], [ch], None, [256], [0, 256]))

    return histogram

def createHisto2(img):

    histogram = [] 
    for ch in range(3):
        histogram.append(np.histogram(img[:,:,ch], bins=256, range=(0,256))[0])

    return histogram

def main():

    argv = sys.argv
    argc = len(argv)

    print('%s creates histogram of a image' % argv[0])
    print('[usage] python %s <image> [<method(scratch/opencv/numpy)]' % argv[0])

    if argc < 2:
        quit()

    img = cv2.imread(argv[1])

    histogram = None

    freq = cv2.getTickFrequency()
    count0 = cv2.getTickCount()
    
    if argc > 2:
        if argv[2] == 'opencv':
            histogram = createHisto1(img)
        elif argv[2] == 'numpy':
            histogram = createHisto2(img)
        else:
            histogram = createHisto0(img)
    else:
        histogram = createHisto0(img)

    count1 = cv2.getTickCount()
    print((count1 - count0) / freq)
    
    plt.plot(histogram[0], color='blue')
    plt.plot(histogram[1], color='green')
    plt.plot(histogram[2], color='red')
    plt.show()

if __name__ == '__main__':
    main()
