import os, sys, cv2
import numpy as np

def main():

    left = 0 # left most x coordinate of the position where foreground is placed
    top = 0 # top most y coordinate of the position where foreground is placed

    argv = sys.argv
    argc = len(argv)

    print('%s blends image using mask' % argv[0])
    print('[usage] python %s <foreground image> <matte image> <background image> [<left pos(0)> <top pos(0)>]' % argv[0])

    if argc < 3:
        quit()

    fore = cv2.imread(argv[1],cv2.IMREAD_COLOR)
    h_fore, w_fore = fore.shape[:2]

    matte = cv2.imread(argv[2],cv2.IMREAD_COLOR)
    h_matte, w_matte = matte.shape[:2]

    if h_matte != h_fore or w_matte != w_fore:
        print('size mismatch (fore: %d x %d, matte: %d x %d)' % (w_fore, h_fore, w_matte, h_matte))
        quit()

    fore = np.array(fore, np.float32) / 255.0
    matte = np.array(matte, np.float32) / 255.0

    back = cv2.imread(argv[3], cv2.IMREAD_COLOR)
    h_back, w_back = back.shape[:2]

    back = np.array(back, np.float32) / 255.0

    if argc > 4:
        left = int(argv[4])

    if argc > 5:
        top = int(argv[5])

    if left >= w_back or top >= h_back:
        print('specified position exceeds background size (back: %d x %d, pos: (%d, %d))' % (w_back, h_back, left, top))
        quit()

    right = min(left + w_fore, w_back)
    w = right - left

    bottom = min(top + h_fore, h_back)
    h = bottom - top

    patch_back = back[top:bottom, left:right]
    patch_fore = fore[0:h, 0:w]
    patch_matte = matte[0:h, 0:w]

    patch = patch_fore * patch_matte + patch_back * (1.0 - patch_matte)

    back[top:bottom, left:right] = patch

    cv2.imshow('matted', back)

    print('Hit s-key to save and terminate')
    print('Hit any other key to quit')

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == ord('s') or key == ord('S'):
        back = np.clip(back *255, 0, 255).astype(np.uint8)
        cv2.imwrite('blend.png', back)
        print('save blend.png')

if __name__ == '__main__':
    main()
