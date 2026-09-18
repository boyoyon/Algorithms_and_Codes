import cv2, os, sys
import numpy as np

def add_gaussian_noise(image, level=25):
    
    # 画像と同じサイズのランダムノイズ（平均0、標準偏差=level）を生成
    mean = 0
    sigma = level
    noise = np.random.normal(mean, sigma, image.shape).astype('int16')
    
    # 画像にノイズを加算し、0〜255の範囲にクリップ（丸め処理）する
    noisy_image = image.astype('int16') + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')
    
    return noisy_image

def main():

    argv = sys.argv
    argc = len(argv)

    print('%s add gaussian noise to the image' % argv[0])
    print('[usage] python %s <image> [<noise level (default: 30)> <number of noisy images(default: 1)>]' % argv[0])

    if argc < 2:
        quit()

    image = cv2.imread(argv[1])

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]

    noise_level = 30

    if argc > 2:
        noise_level = float(argv[2])

    nr_images = 1
 
    if argc > 3:
        nr_images = int(argv[3])

    for i in range(nr_images):

        result = add_gaussian_noise(image, level = noise_level)
        dst_path = '%s_noisy_%04d.png' % (filename, i+1)
        cv2.imwrite(dst_path, result)
        print('save %s' % dst_path)

if __name__ == '__main__':
    main()
