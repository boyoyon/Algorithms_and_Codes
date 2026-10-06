import cv2, os, sys
import numpy as np


def wiener_filter(blurred_img, kernel, K=0.01):
    """ウィーナーフィルターによるブレ除去処理

    Args:
        blurred_img (numpy.ndarray): ブレ画像 (グレースケール, 0-255)
        kernel (numpy.ndarray): カーネル画像 (グレースケール)
        K (float): ノイズパラメータ (SNRの逆数に相当。小さいほど鮮鋭化、大きいほどノイズ抑制)

    Returns:
        numpy.ndarray: 復元画像 (uint8)
    """
    # 1. 浮動小数点数型(0.0 - 1.0)に正規化
    blurred_f = blurred_img.astype(np.float64) / 255.0
    kernel_f = kernel.astype(np.float64)

    # 2. カーネルの合計値を1に正規化（明るさの変化を防ぐ）
    if np.sum(kernel_f) > 0:
        kernel_f /= np.sum(kernel_f)

    # 3. カーネル画像をブレ画像と同じサイズにパディングし、中心を原点にシフト
    kh, kw = kernel_f.shape
    bh, bw = blurred_f.shape
    padded_kernel = np.zeros((bh, bw))
    padded_kernel[:kh, :kw] = kernel_f
    # 周波数領域での位相ズレを防ぐため、カーネルの中心を左上にロール(シフト)する
    padded_kernel = np.roll(padded_kernel, -kh // 2, axis=0)
    padded_kernel = np.roll(padded_kernel, -kw // 2, axis=1)

    # 4. 2次元高速フーリエ変換 (FFT)
    Blurred_FFT = np.fft.fft2(blurred_f)
    Kernel_FFT = np.fft.fft2(padded_kernel)

    # 5. ウィーナーフィルターの数式を適用
    # G(u,v) = conj(H(u,v)) / (|H(u,v)|^2 + K)
    Kernel_FFT_conj = np.conj(Kernel_FFT)
    Wiener_Filter = Kernel_FFT_conj / (np.abs(Kernel_FFT) ** 2 + K)

    # 6. フィルタリングと逆フーリエ変換 (IFFT)
    Restored_FFT = Blurred_FFT * Wiener_Filter
    restored_f = np.fft.ifft2(Restored_FFT)
    restored_f = np.real(restored_f)

    # 7. ピクセル値を0〜1にクリップして255階調に戻す
    restored_f = np.clip(restored_f, 0.0, 1.0)
    return (restored_f * 255).astype(np.uint8)

def main():

    argv = sys.argv
    argc = len(argv)

    print('%s deblurs the image by wiener filter' % argv[0])
    print('[usage] python %s <blurred image> <kernel image>' % argv[0])

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

    cv2.imshow('blurred', blurred)

    b, g, r = cv2.split(blurred)

    deblurred_b = wiener_filter(b, kernel, K=0.01)
    deblurred_g = wiener_filter(g, kernel, K=0.01)
    deblurred_r = wiener_filter(r, kernel, K=0.01)

    deblurred = cv2.merge([deblurred_b, deblurred_g, deblurred_r])

    cv2.imshow('deblurred', deblurred)

    print('Hit s-key to save and terminate')
    print('Hit any other key to quit')

    key = cv2.waitKey(0)
    cv2.destroyAllWindows() 

    if key == ord('s') or key == ord('S'):

        base = os.path.basename(argv[1])
        filename = os.path.splitext(base)[0]
        dst_path = '%s_deblur_by_wiener_filter.png' % filename

        cv2.imwrite(dst_path, deblurred)
        print('save %s' % dst_path)

if __name__ == "__main__":
    main()


