import cv2, os, sys
import numpy as np

sigma = 80.0

def replace_zero_with_min(channel):
    """対数計算(log)でのゼロエラー（無限大）を防ぐため、0を極小値に置き換える"""
    min_nonzero = np.min(channel[channel > 0]) if np.any(channel > 0) else 1
    channel[channel == 0] = min_nonzero
    return channel

def single_scale_retinex(img_v, sigma):
    """SSR (Single-Scale Retinex) の処理"""
    # ゼロ埋め対策
    img_v = replace_zero_with_min(img_v.astype(np.float64))
    
    # ガウシアンフィルタによる周囲環境光（ぼかし画像）の推定
    # ※ sigma=0 の時、ksizeから自動計算されるよう第3引数に設定
    blur = cv2.GaussianBlur(img_v, (0, 0), sigma)
    blur = replace_zero_with_min(blur)
    
    # 数式: log(I) - log(F * I)
    ssr = np.log10(img_v) - np.log10(blur)
    return ssr

def multi_scale_retinex(img_v, sigmas, weights):
    """MSR (Multi-Scale Retinex) の処理"""
    msr = np.zeros(img_v.shape, dtype=np.float64)
    
    # 複数のSSRを重み付けして足し合わせる
    for sigma, weight in zip(sigmas, weights):
        msr += weight * single_scale_retinex(img_v, sigma)
        
    return msr

def restore_and_normalize(retinex_v):
    """Retinex処理後のデータを0-255の範囲に正規化してuint8に変換する"""
    # 平均と標準偏差を使って、2シグマの範囲を0-255にマッピング（コントラスト調整）
    mean = np.mean(retinex_v)
    std = np.standard(retinex_v) if hasattr(np, 'standard') else np.std(retinex_v)
    
    min_val = mean - 2 * std
    max_val = mean + 2 * std
    
    normalized = (retinex_v - min_val) / (max_val - min_val) * 255
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return normalized

def main():

    global sigma

    argv = sys.argv
    argc = len(argv)

    print('%s executes Retinex' % argv[0])
    print('[usage] python %s <image> [<sigma(%.1f)>]' % (argv[0], sigma))

    if argc < 2:
        quit()

    # 画像の読み込み (BGR)
    img = cv2.imread(argv[1])
    if img is None:
        print('failed to load image: %s' % argv[1])
        quit()
    
    cv2.imshow('original', img)

    # HSV空間に変換してV（明度）チャンネルを抽出
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    if argc > 2:
        sigma = float(argv[2])

    # 1. SSRの実行
    ssr_raw = single_scale_retinex(v, sigma=sigma)
    ssr_v = restore_and_normalize(ssr_raw)
    img_ssr = cv2.cvtColor(cv2.merge([h, s, ssr_v]), cv2.COLOR_HSV2BGR)

    cv2.imshow('SSR', img_ssr)

    # 2. MSRの実行 (小・中・大のスケール、均等な重み)
    sigmas = [15/80*sigma, sigma, 250/80*sigma]
    weights = [1/3, 1/3, 1/3]
    msr_raw = multi_scale_retinex(v, sigmas, weights)
    msr_v = restore_and_normalize(msr_raw)
    img_msr = cv2.cvtColor(cv2.merge([h, s, msr_v]), cv2.COLOR_HSV2BGR)

    cv2.imshow('MSR', img_msr)

    print('Hit s-key to save and terminate')
    print('Hit any other key to quit')

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == ord('s') or key == ord('S'):

        base = os.path.basename(argv[1])
        filename = os.path.splitext(base)[0]

        SIGMA1 = '%.0f' % (15/80*sigma)
        SIGMA2 = '%.0f' % sigma
        SIGMA3 = '%.0f' % (250/80*sigma)

        ssr_path = '%s_ssr_%s.png' % (filename, SIGMA2)
        cv2.imwrite(ssr_path, img_ssr)
        print('save %s' % ssr_path)

        msr_path = '%s_msr_%s_%s_%s.png' % (filename, SIGMA1, SIGMA2, SIGMA3)
        cv2.imwrite(msr_path, img_msr)
        print('save %s' % msr_path)

if __name__ == '__main__':
    main()
