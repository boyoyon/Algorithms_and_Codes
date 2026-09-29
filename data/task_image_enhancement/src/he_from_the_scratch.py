import cv2, os, sys
import numpy as np

def he(channel):

    h = channel.shape[0]
    w = channel.shape[1]
    nr_pixels = h * w

    equalized = np.zeros((h,w), np.uint8)

    equivs = []
    for i in range(256):
        equivs.append(np.where(channel == i))

    sum = 0

    for i in range(256):
        sum += len(equivs[i][0])
        equalized[equivs[i]] = 255 * sum // nr_pixels

    return equalized

argv= sys.argv
argc = len(argv)

if argc < 2:
    print('%s equalizes histogram from the scratch' % argv[0])
    print('%s <image>' % argv[0])
    quit()

src = cv2.imread(argv[1])
height, width = src.shape[:2]
nrPixels = height * width

b, g, r = cv2.split(src)

he_b = he(b)
he_g = he(g)
he_r = he(r)

dst = cv2.merge([he_b, he_g, he_r])

cv2.imshow('src', src)
cv2.imshow('dst', dst)

print('Hit s-key to save and terminate')
print('Hit any other key to quit')

key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_he_from_the_scratch.png' % filename

    cv2.imwrite(dst_path, dst)
    print('save %s' % dst_path)

cv2.destroyAllWindows()
