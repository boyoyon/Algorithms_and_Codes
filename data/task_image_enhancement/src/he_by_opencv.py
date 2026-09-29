import cv2, sys

argv= sys.argv
argc = len(argv)

if argc < 2:
    print('%s equalizes histogram using opencv' % argv[0])
    print('[usage] python %s <image>' % argv[0])
    quit()

src = cv2.imread(argv[1])
b, g, r = cv2.split(src)
b = cv2.equalizeHist(b)
g = cv2.equalizeHist(g)
r = cv2.equalizeHist(r)
dst = cv2.merge((b, g, r))

cv2.imshow('src', src)
cv2.imshow('dst', dst)

print('Hit s-key to save and terminate')
print('Hit any other key to quit')

key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_he_by_opencv.png' % filename

    cv2.imwrite(dst_path, dst)
    print('save %s' % dst_path)

cv2.destroyAllWindows()


