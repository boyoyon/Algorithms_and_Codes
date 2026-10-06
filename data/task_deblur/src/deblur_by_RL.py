import cv2, os, sys
import numpy as np
from skimage import restoration

ITER = 30

def richardson_lucy_deblur(blurred_img, kernel, num_iter=30):
    """Richardson-Lucy アルゴリズムによるブレ除去処理

    Args:
        blurred_img (numpy.ndarray): ブレ画像 (グレースケール, 0-255)
        kernel (numpy.ndarray): カーネル画像 (グレースケール)
        num_iter (int): 反復回数 (多いほど鮮鋭化するが、ノイズも増加する)

    Returns:
        numpy.ndarray: 復元画像 (uint8)
    """
    # 1. 浮動小数点数型(0.0 - 1.0)に正規化 (scikit-imageの要件)
    blurred_f = blurred_img.astype(np.float64) / 255.0
    kernel_f = kernel.astype(np.float64)

    # 2. カーネルの合計値を1に正規化
    if np.sum(kernel_f) > 0:
        kernel_f /= np.sum(kernel_f)

    # 3. Richardson-Lucy デコンボリューションの実行
    # 内部で畳み込み(FFTベース)と反復処理が自動で行われます
    restored_f = restoration.richardson_lucy(
        blurred_f, kernel_f, num_iter=num_iter
    )

    # 4. ピクセル値を0〜1にクリップして255階調に戻す
    restored_f = np.clip(restored_f, 0.0, 1.0)
    return (restored_f * 255).astype(np.uint8)

def main():

    global ITER

    argv = sys.argv
    argc = len(argv)

    print('%s deblurs the image by RL algorithm' % argv[0])
    print('[usage] python %s <blurred image> <kernel image> [<No. of iteration(%d)>]' % (argv[0], ITER))

    if argc < 3:
        quit()

    blurred = cv2.imread(argv[1])

    if blurred is None:
        print('failed to load blurred image: %s' % argv[1])
        quit()

    kernel = cv2.imread(argv[2], cv2.IMREAD_GRAYSCALE)

    if blurred is None:
        print('failed to load kernel image: %s' % argv[2])
        quit()

    if argc > 3:
        ITER = int(argv[3])

    cv2.imshow('blurred', blurred)

    b, g, r = cv2.split(blurred)

    deblurred_b = richardson_lucy_deblur(b, kernel, num_iter=ITER)
    deblurred_g = richardson_lucy_deblur(g, kernel, num_iter=ITER)
    deblurred_r = richardson_lucy_deblur(r, kernel, num_iter=ITER)

    deblurred = cv2.merge([deblurred_b, deblurred_g, deblurred_r])

    cv2.imshow('deblurred', deblurred)

    print('Hit s-key to save and terminate')
    print('Hit any other key to quit')

    key = cv2.waitKey(0)
    cv2.destroyAllWindows() 

    if key == ord('s') or key == ord('S'):

        base = os.path.basename(argv[1])
        filename = os.path.splitext(base)[0]
        dst_path = '%s_deblur_by_RL.png' % filename

        cv2.imwrite(dst_path, deblurred)
        print('save %s' % dst_path)

if __name__ == "__main__":
    main()


