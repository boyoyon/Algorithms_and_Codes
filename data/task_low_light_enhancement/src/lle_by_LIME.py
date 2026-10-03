import cv2, os, sys
import numpy as np
from LIME import LIME

iter = 2
alpha = 5
rho = 10
gamma = 0.7

def main():

    global iter, alpha, rho, gamma

    argv = sys.argv
    argc = len(argv)

    print('%s executes LLE by LIME' % argv[0])
    print('[usage] python %s <image> [<iter(%d)> <alpha(%.2f)> <rho(%.2f)> <gamma(%.2f)>]' % (argv[0], iter, alpha, rho, gamma))

    if argc < 2:
        quit()

    if argc > 2:
        iter = int(argv[2])

    if argc > 3:
        alpha = float(argv[3])

    if argc > 4:
        rho = float(argv[4])

    if argc > 5:
        gamma = float(argv[5])

    lime = LIME(iterations=iter, alpha=alpha, rho=rho, gamma=gamma, strategy=2, exact=True)
    lime.load(argv[1])
    dst = lime.run()

    dst = np.clip(dst, 0, 255).astype(np.uint8)
    cv2.imshow('LLE', dst)

    print('Hit s-key to save and terminate.')
    print('Hit any other key to quit.')

    key = cv2.waitKey(0)

    cv2.destroyAllWindows()

    if key == ord('s') or key == ord('S'):
        base = os.path.basename(argv[1])
        filename = os.path.splitext(base)[0]

        ITER = '%d' % iter
        ALPHA = ('%.2f' % alpha).replace('.', '-')
        RHO = ('%.2f' % rho).replace('.', '-')
        GAMMA = ('%.2f' % gamma).replace('.', '-')

        dst_path = '%s_lime_%s_%s_%s_%s.png' % (filename, ITER, ALPHA, RHO, GAMMA) 
        cv2.imwrite(dst_path, dst)
        print('save %s' % dst_path)

if __name__ == "__main__":
    main()
