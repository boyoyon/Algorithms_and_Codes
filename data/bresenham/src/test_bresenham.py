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

def bresenham(screen, xs, ys, xe, ye, col):

    dx = xe - xs
    dy = ye - ys

    b = col[0]
    g = col[1]
    r = col[2]

    if dx == 0 and dy == 0:
        return

    elif  dx == 0:

        if ye > ys:

            for y in range(ys, ye+1):

                screen[y][xs][0] = b
                screen[y][xs][1] = g
                screen[y][xs][2] = r
            
        else:

            for y in range(ye, ys+1):

                screen[y][xs][0] = b
                screen[y][xs][1] = g
                screen[y][xs][2] = r

    elif dy == 0:

        if xe > xs:

            for x in range(xs, xe+1):
 
                screen[ys][x][0] = b
                screen[ys][x][1] = g
                screen[ys][x][2] = r

        else:

            for x in range(xe, xs+1):
 
                screen[ys][x][0] = b
                screen[ys][x][1] = g
                screen[ys][x][2] = r

    else:

        x = xs
        y = ys

        x_step = 1
        if xs > xe:
            x_step = -1

        y_step = 1
        if ys > ye:
            y_step = -1

        a = np.abs(ye - ys) * 2
        th = np.abs(xs - xe) * -1
        th_step = np.abs(xe - xs) * -2


        screen[y][x][0] = b
        screen[y][x][1] = g
        screen[y][x][2] = r

        while x != xe:

            x += x_step
            th += a

            while th > 0:

                y += y_step
                th += th_step

                screen[y][x][0] = b
                screen[y][x][1] = g
                screen[y][x][2] = r

            screen[y][x][0] = b  
            screen[y][x][1] = g
            screen[y][x][2] = r

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

        key = cv2.waitKey(100)

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

            bresenham(screen, xs, ys, xe, ye, colors[c])
            cv2.imshow('screen', screen)
 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()