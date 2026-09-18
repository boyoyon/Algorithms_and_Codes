import cv2, sys
import numpy as np
import matplotlib.pyplot as plt

def createDiff(img):

    img = img.astype(np.int32)
    H, W = img.shape[:2]

    diffX = img[:,0:W-1,:] - img[:,1:W,:]
    diffX = diffX.flatten()

    diffY = img[0:H-1,:,:] - img[1:H,:,:]
    diffY = diffY.flatten()

    merged = np.concatenate((diffX, diffY))

    return merged

def main():

    argv = sys.argv
    argc = len(argv)
    
    print('%s creates histogram of image gradient' % argv[0])
    print('[usage] python %s <image file>' % argv[0])
    
    if argc < 2:
        quit()
    
    img = cv2.imread(argv[1])
    diff = createDiff(img)
   
    print(np.min(diff), np.max(diff))

    hist = np.histogram(diff, bins=255*2+1, range=(-255,255))[0]
    plt.plot(hist)
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    main()