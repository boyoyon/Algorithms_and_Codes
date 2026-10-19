import os, sys, cv2
import numpy as np

color = (0, 255,0)
thickness = 1

ESC_KEY = 27

# keycode for cv2.waitKeyEx()
LEFT  = 2424832
UP    = 2490368
RIGHT = 2555904
DOWN  = 2621440

font = cv2.FONT_HERSHEY_PLAIN
font_size = 2
font_color = (0, 255, 0)
font_pos = (10, 30)
font_pos2 = (10, 60)

WIDTH = 1920
HEIGHT = 1080

G_SIZE_MAX = 31
G_SIZE_MIN = 3
G_SIZE = G_SIZE_MIN

def AlphaBlend(img1, img2, alpha):
    dst = alpha * img1 + (1.0 - alpha) * img2
    return dst

def usage(progName):
    print('%s simulates unsharp masking' % progName)
    print('%s <image>' % progName)

def main():

    global mouse_x, mouse_y, state, G_SIZE

    argv = sys.argv
    argc = len(argv)

    usage(argv[0])
    
    if argc < 2:
        quit()
   
    SCALE = 1.0
    prevSCALE = -1.0

    ALPHA = 3.0
    prevALPHA = -1.0

    prevG_SIZE = -1

    src1 = cv2.imread(argv[1])
    H1, W1 = src1.shape[:2]

    if W1 > WIDTH:
        SCALE = WIDTH / W1

    src2 = cv2.GaussianBlur(src1, (G_SIZE, G_SIZE), 0)

    src1 = src1.astype(np.float32) / 255.0
    src2 = src2.astype(np.float32) / 255.0

    print('Hit +/- to up/down scale')
    print('Hit arrow-key to up/down alpha')
    print('Hit G-key to up Gaussian size')
    print('Hit g-key to down Gaussian size')
    print('Hit s-key to save result image')
    print('hit ESC-key to terminate')

    w = int(W1 * SCALE)
    h = int(H1 * SCALE)

    clone1 = cv2.resize(src1, (w, h))
    clone2 = cv2.resize(src2, (w, h))

    cv2.imshow('source1', clone1)
    
    fUPDATE = True
    fSAVE = False

    while True:

        if ALPHA != prevALPHA:
            fUPDATE = True
            prevALPHA = ALPHA


        if SCALE != prevSCALE:
            w = int(W1 * SCALE)
            h = int(H1 * SCALE)
            clone1 = cv2.resize(src1, (w, h))
            clone2 = cv2.resize(src2, (w, h))
            fUPDATE = True
            prevSCALE = SCALE

        if G_SIZE != prevG_SIZE:
            src2 = cv2.GaussianBlur(src1, (G_SIZE, G_SIZE),0)
            clone2 = cv2.resize(src2, (w, h))
            fUPDATE = True
            prevG_SIZE = G_SIZE

        if fUPDATE:
            dst = AlphaBlend(clone1, clone2, ALPHA)
            cv2.putText(dst, 'alpha: %.1f' % ALPHA, 
                font_pos, font, font_size, font_color, 2)
            cv2.putText(dst, 'gauss: %d' % G_SIZE,
                font_pos2, font, font_size, font_color, 2)
            cv2.imshow('blend', dst)
            fUPDATE = False

        key = cv2.waitKeyEx(10)
    
        if key == LEFT:
            ALPHA += 0.1
        
        elif key == UP:
            ALPHA += 0.3

        elif key == RIGHT:
            ALPHA -= 0.1

        elif key == DOWN:
            ALPHA -= 0.3

        elif key == ord('-'):
            SCALE *= 0.9

        elif key == ord('+'):
            SCALE *= 1.1

        elif key == ord('G'):
            G_SIZE += 2
            if G_SIZE > G_SIZE_MAX:
                G_SIZE = G_SIZE_MAX

        elif key == ord('g'):
            G_SIZE -= 2
            if G_SIZE < G_SIZE_MIN:
                G_SIZE = G_SIZE_MIN

        elif key == ord('s') or key == ord('S'):
            fSAVE = True
            break

        elif key == ESC_KEY:
            break

    cv2.destroyAllWindows()

    if fSAVE:
        dst = AlphaBlend(src1, src2, ALPHA)
        dst *= 255.0
        dst = np.clip(dst, 0, 255)
        dst = dst.astype(np.uint8)

        base = os.path.basename(argv[1])
        filename = os.path.splitext(base)[0]
        strALPHA = ('%.1f' % ALPHA).replace('.', '_')
        dst_path = '%s_%d_%s.png' % (filename, G_SIZE, strALPHA)
        cv2.imwrite(dst_path, dst)
        print('save %s' % dst_path)

if __name__ == '__main__':

    main()
