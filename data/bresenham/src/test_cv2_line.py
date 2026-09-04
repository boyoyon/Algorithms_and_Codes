import numpy as np
import cv2

SCREEN_WIDTH = 768
SCREEN_HEIGHT = 512

colors = [[ 64,  64,  64], 
          [  0,  0,  255],
          [  0, 255,   0],
          [  0, 255, 255],
          [255,   0,   0],
          [255,   0, 255],
          [255, 255,   0],
          [255, 255, 255]]

ESC = 27

def main():

    screen = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8)
    cv2.imshow('screen', screen)

    print('Hit ESC key to terminate')
    print('Hit any other key to pause and restart')

    key = -1
    fRun = True
    rng = np.random.default_rng()
    nr_colors = len(colors)

    while key != ESC:

        key = cv2.waitKey(10)

        if key == ESC:
            break
        elif key != -1:
            fRun = not fRun

        if fRun:
            xs = rng.integers(0, SCREEN_WIDTH)
            ys = rng.integers(0, SCREEN_HEIGHT)
            xe = rng.integers(0, SCREEN_WIDTH)
            ye = rng.integers(0, SCREEN_HEIGHT)
            c = rng.integers(0, nr_colors)

            cv2.circle(screen,(xs, ys), 3, colors[c], -1)
            cv2.circle(screen,(xe, ye), 3, colors[c], -1)

            cv2.line(screen, (xs, ys), (xe, ye), colors[c], 1)
            cv2.imshow('screen', screen)
 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()