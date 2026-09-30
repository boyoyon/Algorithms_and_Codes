import numpy as np
import cv2, os, sys

ESC_KEY = 27
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1024
SCALE = 1.0

def callback(x):
    pass # do nothing

def display_cdf(image):

    CDF = np.ones((256,256,3), np.uint8)
    CDF *= 255

    b, g, r = cv2.split(image)
    H, W = image.shape[:2]

    hist,bins = np.histogram(b.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_b = cdf * 255 / (H*W)
    cdf_b = cdf_b.astype(np.uint8)

    for i in range(1, 256):
        cv2.line(CDF, (i-1, 255 - cdf_b[i-1]), (i, 255 - cdf_b[i]), (255, 0, 0), 1)

    hist,bins = np.histogram(g.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_g = cdf * 255 / (H*W)
    cdf_g = cdf_g.astype(np.uint8)

    for i in range(1, 256):
        cv2.line(CDF, (i-1, 255 - cdf_g[i-1]), (i, 255 - cdf_g[i]), (0, 255, 0), 1)

    hist,bins = np.histogram(r.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_r = cdf * 255 / (H*W)
    cdf_r = cdf_r.astype(np.uint8)

    for i in range(1, 256):
        cv2.line(CDF, (i-1, 255 - cdf_r[i-1]), (i, 255 - cdf_r[i]), (0, 0, 255), 1)

    cv2.imshow('CDF', CDF)

argv = sys.argv
argc = len(argv)

print('%s interpolates histogram between original and histogram equalized' % argv[0])
print('[usage] python %s <image>' % argv[0])

if argc < 2:
    quit()

src = cv2.imread(argv[1])
if src is None:
    print('failed to load image %s' % argv[1])
    quit()

src_clone = src.copy()
cv2.imshow('original', src_clone)

H, W = src.shape[:2]
nr_pixels = H * W

channels = cv2.split(src)

# store equivalent pixels position
equivs = []
for i in range(3):
    equiv = []
    for j in range(256):
        equiv.append(np.where(channels[i] == j))

    equivs.append(equiv)

# store HE values
hes = []
for i in range(3):
    he = []
    sum = 0
    for j in range(256):
        # add the number of y-coordinates = histogram
        sum += len(equivs[i][j][0])
        he.append(255 * sum // nr_pixels)

    hes.append(he)

dst = src.copy()
cv2.imshow('dst', dst)

blue = 50
green = 50
red = 50

prev_blue = -1
prev_green = -1
prev_red = -1

# create trackbar
cv2.createTrackbar('blue',   'dst', 1, 200, callback)
cv2.setTrackbarPos('blue', 'dst', blue+50)

cv2.createTrackbar('green', 'dst', 1, 200, callback)
cv2.setTrackbarPos('green', 'dst', green+50)

cv2.createTrackbar('red',     'dst', 1, 200, callback)
cv2.setTrackbarPos('red', 'dst', red+50)

dst_channels = cv2.split(dst)

print('Hit ESC key to quit')
print('Hit s key to save and terminate')
print('Hit + key to scale up the image')
print('Hit - key to scale down the image')

fRatio = True
fScale = True
prev_scale = -1

while True:

    blue = cv2.getTrackbarPos('blue', 'dst') - 50
    green = cv2.getTrackbarPos('green', 'dst') - 50
    red = cv2.getTrackbarPos('red', 'dst') - 50

    key = cv2.waitKey(10)

    if key == ESC_KEY or key == ord('s') or key == ord('S'):
        break
    
    elif key == ord('+'):
        SCALE *= 1.1

    elif key == ord('-'):
        SCALE *= 0.9

    if blue != prev_blue or green != prev_green or red != prev_red:

        if blue != prev_blue:

            for j in range(256):
                new_value = (blue * j + (100 - blue) * hes[0][j]) // 100
                dst_channels[0][equivs[0][j]] = np.clip(new_value, 0, 255)

            prev_blue = blue

        if green != prev_green:

            for j in range(256):
                new_value = (green * j + (100 - green) * hes[1][j]) // 100
                dst_channels[1][equivs[1][j]] = np.clip(new_value, 0, 255)

            prev_green = green

        if red != prev_red:

            for j in range(256):
                new_value = (red * j + (100 - red) * hes[2][j]) // 100
                dst_channels[2][equivs[2][j]] = np.clip(new_value, 0, 255)

            prev_red = red

        dst = cv2.merge(dst_channels)
        fRatio = True

    if SCALE != prev_scale:
        fScale = True
        prev_scale = SCALE

    if fRatio or fScale:
        if SCALE != 1.0:

            w = int(W * SCALE)
            h = int(H * SCALE)
            dst_clone = cv2.resize(dst, (w, h))
            src_clone = cv2.resize(src, (w, h))
            cv2.imshow('original', src_clone)

            if fScale:
                cv2.destroyWindow('dst')
                cv2.imshow('dst', dst_clone)
                cv2.createTrackbar('blue',   'dst', 1, 200, callback)
                cv2.setTrackbarPos('blue', 'dst', blue+50)
                cv2.createTrackbar('green', 'dst', 1, 200, callback)
                cv2.setTrackbarPos('green', 'dst', green+50)
                cv2.createTrackbar('red',     'dst', 1, 200, callback)
                cv2.setTrackbarPos('red', 'dst', red+50)
            else:
                cv2.imshow('dst', dst_clone)

        else:
            dst_clone = dst.copy()
            cv2.imshow('dst', dst_clone)

        if fRatio:
            display_cdf(dst)
            fRatio = False

        if fScale:
            fScale= False

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_b%d_g%d_r%d.png' % (filename, blue+50, green+50, red+50)
    cv2.imwrite(dst_path, dst)
    print('save %s' % dst_path)

cv2.destroyAllWindows()