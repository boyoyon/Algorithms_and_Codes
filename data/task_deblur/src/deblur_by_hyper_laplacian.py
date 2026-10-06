import cv2, os, sys
import numpy as np

def solve_xi_alpha_05(v, beta, lmbda):
    """alpha = 1/2 のときの補助変数 w の各ピクセル解を解析的に（3次方程式の根として）解く

    |w|^0.5 + (beta/2) * (w - v)^2 を最小化する w を求める
    """
    # 論文記載の解析解（3次方程式の公式）の簡易実装
    # 符号を退避して絶対値で計算
    sign_v = np.sign(v)
    v_abs = np.abs(v)

    # 方程式の係数など
    c1 = -v_abs
    c2 = 1.0 / beta  # lambda=1 の前提（あるいはあらかじめ v や beta に反映）

    # 3次方程式の判別式チェック用
    # 論文中の閾値判定（w=0になるか、正の根を持つか）
    threshold = (3.0 / 4.0) * (c2 ** (2.0 / 3.0))

    # 出力配列の初期化 (デフォルトは 0)
    w = np.zeros_like(v)

    # 閾値を超えたピクセルのみ根の計算を行う
    mask = v_abs > threshold
    if np.any(mask):
        vm = v_abs[mask]
        # 3次方程式の公式の変形
        # z^3 - v*z + 1/(2*beta) = 0  (ここで z = sqrt(w))
        # 論文の手順に沿った最も実数解に近い解を選択
        phi = np.arccos((1.0 / (4.0 * beta)) * ((3.0 / vm) ** 1.5))
        z = 2.0 * np.sqrt(vm / 3.0) * np.cos(phi / 3.0)
        w[mask] = (z**2) * sign_v[mask]

    return w


def hyper_laplacian_deblur(blurred, kernel, lmbda=3000, max_iter=5):
    """Fast Image Deconvolution using Hyper-Laplacian Priors (alpha=0.5)"""
    h, w = blurred.shape
    kh, kw = kernel.shape

    # 1. 画像とカーネルを0.0-1.0に正規化
    x = blurred.astype(np.float64) / 255.0
    y = x.copy()  # 初期値はブレ画像

    k = kernel.astype(np.float64)
    if np.sum(k) > 0:
        k /= np.sum(k)

    # 2. カーネルを画像サイズにパディングしてFFT用にシフト
    k_padded = np.zeros((h, w))
    k_padded[:kh, :kw] = k
    #k_padded = k_padded.roll(-kh // 2, axis=0)
    #k_padded = k_padded.roll(-kw // 2, axis=1)

    K_FFT = np.fft.fft2(k_padded)
    K_FFT_conj = np.conj(K_FFT)
    K_abs_sq = np.abs(K_FFT) ** 2
    Y_FFT = np.fft.fft2(y)

    # 3. 差分フィルタ（微分オペレータ）の準備
    dx = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float64)
    dy = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float64)

    dx_padded = np.zeros((h, w))
    dx_padded[:3, :3] = dx
    #dx_padded = dx_padded.roll(-1, axis=0).roll(dx_padded, -1, axis=1)
    Dx_FFT = np.fft.fft2(dx_padded)

    dy_padded = np.zeros((h, w))
    dy_padded[:3, :3] = dy
    #dy_padded = np.roll(dy_padded, -1, axis=0).roll(dy_padded, -1, axis=1)
    Dy_FFT = np.fft.fft2(dy_padded)

    Dx_abs_sq = np.abs(Dx_FFT) ** 2
    Dy_abs_sq = np.abs(Dy_FFT) ** 2
    Denom_D = Dx_abs_sq + Dy_abs_sq

    # 4. 継続条件用のベータ（継続反復で大きくしていくペナルティ係数）
    # 論文のスケジュールに従い、徐々に拘束を強くする
    beta_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # メインループ (Alternating Minimization)
    for beta in beta_list:
        # --- Sub-problem 1: w-step (各ピクセル独立の非凸問題) ---
        # 今の画像 x の勾配を計算
        x_fft = np.fft.fft2(x)
        grad_x = np.real(np.fft.ifft2(x_fft * Dx_FFT))
        grad_y = np.real(np.fft.ifft2(x_fft * Dy_FFT))

        # 3次方程式の解（LUTの代わりに解析解）で高速に次の w を決定
        wx = solve_xi_alpha_05(grad_x, beta, lmbda)
        wy = solve_xi_alpha_05(grad_y, beta, lmbda)

        # --- Sub-problem 2: x-step (FFTで一発で解けるL2問題) ---
        # 論文の主方程式を周波数領域で解く
        Num = lmbda * K_FFT_conj * Y_FFT + beta * (
            np.conj(Dx_FFT) * np.fft.fft2(wx)
            + np.conj(Dy_FFT) * np.fft.fft2(wy)
        )
        Denom = lmbda * K_abs_sq + beta * Denom_D

        x = np.real(np.fft.ifft2(Num / Denom))

    # 5. 後処理
    x = np.clip(x, 0.0, 1.0)
    return (x * 255).astype(np.uint8)

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

    deblurred_b = hyper_laplacian_deblur(b, kernel, lmbda=2000)
    deblurred_g = hyper_laplacian_deblur(g, kernel, lmbda=2000)
    deblurred_r = hyper_laplacian_deblur(r, kernel, lmbda=2000)

    deblurred = cv2.merge([deblurred_b, deblurred_g, deblurred_r])

    cv2.imshow('deblurred', deblurred)

    print('Hit s-key to save and terminate')
    print('Hit any other key to quit')

    key = cv2.waitKey(0)
    cv2.destroyAllWindows() 

    if key == ord('s') or key == ord('S'):

        base = os.path.basename(argv[1])
        filename = os.path.splitext(base)[0]
        dst_path = '%s_deblur_by_hyper_laplacian.png' % filename

        cv2.imwrite(dst_path, deblurred)
        print('save %s' % dst_path)

if __name__ == "__main__":
    main()


