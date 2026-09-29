import cv2, os, sys

argv= sys.argv
argc = len(argv)

if argc < 2:
    print('%s equalizes histogram using opencv' % argv[0])
    print('%s <image>' % argv[0])
    quit()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

src = cv2.imread(argv[1])
b, g, r = cv2.split(src)
b = clahe.apply(b)
g = clahe.apply(g)
r = clahe.apply(r)
dst = cv2.merge((b, g, r))

cv2.imshow('src', src)
cv2.imshow('dst', dst)

print('Hit s-key to save and terminate')
print('Hit any other key to quit')

key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_clahe_by_opencv.png' % filename

    cv2.imwrite(dst_path, dst)
    print('save %s' % dst_path)

cv2.destroyAllWindows()


