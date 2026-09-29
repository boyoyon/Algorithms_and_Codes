import cv2, sys
import numpy as np
import matplotlib.pyplot as plt

def main():

    argv = sys.argv
    argc = len(argv)

    print('%s creates histogram of a image' % argv[0])
    print('[usage] python %s <image> [<method(scratch/opencv/numpy)]' % argv[0])

    if argc < 2:
        quit()

    img = cv2.imread(argv[1])

    hist,bins = np.histogram(img.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

if __name__ == '__main__':
    main()