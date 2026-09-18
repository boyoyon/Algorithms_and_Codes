import cv2, glob, sys
import numpy as np

argv = sys.argv
argc = len(argv)

print('%s averages images' % argv[0])
print('[usage] python %s <wildcard for images>' % argv[0])

if argc < 2:
    quit()

paths = glob.glob(argv[1])

nr_images = 0
images = None
W = -1
H = -1

for path in paths:

    print('processing %s' % path)

    img = cv2.imread(path).astype(np.float32) / 255.0

    if img is None:
        print('failed to load %s' % path)
        print('skip')
        continue

    h, w = img.shape[:2]
    nr_images += 1

    if images is None:
        H, W = img.shape[:2]
        images = np.zeros((H, W, 3), np.float32)

    if h != H or w != W:
        img = cv2.resize(img, (W, H))

    images += img

images *= 255.0 / nr_images
images = np.clip(images, 0, 255)
images = images.astype(np.uint8)

cv2.imwrite('average.png', images)
print('save average.png')


    


 
