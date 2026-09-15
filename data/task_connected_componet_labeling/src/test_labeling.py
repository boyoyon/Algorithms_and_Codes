import cv2, sys
import numpy as np
from hsv2rgb import hsv2rgb

TH = 64

def create_colors(nrColors):
    colors = []
    step = 360 // nrColors
    for i in range(nrColors):
        H = step * i
        S = 200
        V = 200

        R, G, B = hsv2rgb(H, S, V)
        colors.append((B, G, R))

    return colors

def main():
    argv = sys.argv
    argc = len(argv)

    print('%s executes connected component lableing' % argv[0])
    print('[usage] python %s <image>' % argv[0])

    if argc < 2:
        quit()

    gray = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
    binImg = np.zeros(gray.shape, gray.dtype)
    bgr = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)

    binImg[gray > TH] = 255
    binImg[gray <= TH] = 0

    nlabels, labels, stats, centroid = cv2.connectedComponentsWithStats(binImg)

    colors = create_colors(nlabels - 1)

    for i in range(1, nlabels):
        Ys, Xs = np.where(labels == i)
        for (x, y) in zip(Xs, Ys):
            bgr[y][x] = colors[i - 1]

    cv2.imshow('bgr', bgr)
    cv2.imwrite('labeling.png', bgr)
    cv2.imshow('gray', gray)
    cv2.imshow('bin', binImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()