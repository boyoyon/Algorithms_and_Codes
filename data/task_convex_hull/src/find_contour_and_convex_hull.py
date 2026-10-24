import cv2, os, sys

TH = 230
MIN_NR_CONTOURS = 5

argv = sys.argv
argc = len(argv)

print('%s finds contour and executes convex hull' % argv[0])
print('[usage] python %s <image> [<threshold (%d)>]' % (argv[0], TH))

if argc < 2:
    quit()

src = cv2.imread(argv[1])
cv2.imshow('source', src)

if argc > 2:
    TH = int(argv[2])

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray[gray >= TH] = 255
gray[gray < TH] = 0
cv2.imshow('binary', gray)

contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

dst = src.copy()

for i in range(len(contours)):

    if len(contours[i]) < MIN_NR_CONTOURS: # ignore too simple contours
        continue

    hull = cv2.convexHull(contours[i])
    dst = cv2.drawContours(dst, [hull], -1, (0, 0, 255), 2)

cv2.imshow('result', dst)

print('Hit s-key to save and terminate')
print('Hit any other key to quit')

key =  cv2.waitKey(0)
cv2.destroyAllWindows()

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_convex_hull.png' % filename
    cv2.imwrite(dst_path, dst)
    print('save %s' % dst_path)