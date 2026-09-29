import numpy as np
import sys, cv2, copy

def usage(progName):
    print('%s equalizes histogram' % progName)
    print('Usgae: >python %s <input image> <output image>' % progName)

def callback(x):
    pass # do nothing

def calcHist(src, histo, indices):

    height, width, nChannels = src.shape[:3]
    planes = cv2.split(src)

    for i in range(nChannels):
        for j in range(256):
             indices[i].append(np.where(planes[i] == j))
             histo[i][j] = len(indices[i][j][0])

def adjust_contrast(src, histo, rgb_min, rgb_max, indices):

    height, width, nChannels = src.shape[:3]

    planes = np.zeros((3, height, width), np.uint8)

    for i in range(nChannels):
        for j in range(256):
            if j <= rgb_min:
                planes[i][indices[i][j]] = 0
            elif j >= rgb_max:
                planes[i][indices[i][j]] = 255
            else:
                span = rgb_max - rgb_min
                offset = j - rgb_min
                target = offset *  255 // span
                planes[i][indices[i][j]] = target

    dst = cv2.merge(planes)

    return dst

def main():

    argv = sys.argv
    argc = len(argv)

    if(argc < 2):
        usage(argv[0])
        quit()

    SLIDER = 'slider' # slider window name

    src = cv2.imread(argv[1])
    cv2.imshow('original', src)

    histo = np.zeros((3,256), np.int32)
    rgb_min = 0
    rgb_max = 255
    prev_min = -1
    prev_max = -1

    indices = [[], [], []]

    calcHist(src, histo, indices)

    # create a window
    cv2.namedWindow('contrast')

    slider = np.zeros((10, 512, 3), np.uint8)
    cv2.imshow(SLIDER, slider)

    # create trackbar
    cv2.createTrackbar('rgb_min', SLIDER, 0, 254, callback)
    cv2.setTrackbarPos('rgb_min', SLIDER, 0)
    cv2.createTrackbar('rgb_max', SLIDER, 1, 255, callback)
    cv2.setTrackbarPos('rgb_max', SLIDER, 255)

    while(1):
        # retrieve the current position of trackbar

        rgb_min = cv2.getTrackbarPos('rgb_min', SLIDER)
        rgb_max = cv2.getTrackbarPos('rgb_max', SLIDER)

        if rgb_max <= rgb_min:
            rgb_max = rgb_min + 1
            cv2.setTrackbarPos('rgb_max', SLIDER, rgb_max)

        # adjust contrast
        if rgb_min != prev_min or rgb_max != prev_max:

            dst = adjust_contrast(src, histo, rgb_min, rgb_max, indices)

            cv2.imshow('contrast', dst)
 
            prev_min = rgb_min
            prev_max = rgb_max

        key = cv2.waitKey(100)
        if(key == 27):
            break

    cv2.destroyAllWindows()

    if(argc > 2):
        cv2.imwrite(argv[2], dst)
    else:
        cv2.imwrite('contrast_%d_%d.png' % (rgb_min, rgb_max), dst)

if __name__ == '__main__':
    main()
