import numpy as np
import cv2, os, sys

h = 30
hColor = 20
templateWindowSize = 9
searchWindowSize = 41

argv = sys.argv
argc = len(argv)

print('%s denoises the image by median filter' % argv[0])
print('[usage] python %s <image> [<h (default:%3d)> <hColor (default:%3d)> <templateWindowSize (default:%3d)> <searchWindowSize (default:%3d)>]' % (argv[0], h, hColor, templateWindowSize, searchWindowSize))

if argc < 2:
    quit()

noisy = cv2.imread(argv[1])
cv2.imshow('noisy', noisy)

if argc > 2:
        h = int(argv[2])

if argc > 3:
    hColor = int(argv[3])

if argc > 4:
    templateWindowSize = int(argv[4])
    if templateWindowSize % 2 == 0:
        templateWindowSize += 1

if argc > 5:
    searchWindowSize = int(argv[5])
    if searchWindowSize % 2 == 0:
        searchWindowSize += 1

denoised = cv2.fastNlMeansDenoisingColored(noisy,None,h, hColor, templateWindowSize, searchWindowSize)

cv2.imshow('denoised', denoised)

print('Hit s-key to save and terminate')
print('Hit any other key to quit')
key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_denosed_by_nlm_%03d_%03d_%03d_%03d.png' % (filename, h, hColor, templateWindowSize, searchWindowSize)
    cv2.imwrite(dst_path, denoised)
    print('save %s' % dst_path)

cv2.destroyAllWindows()