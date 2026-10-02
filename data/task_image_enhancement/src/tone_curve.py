import cv2, os, sys
import numpy as np

def display_tones(tones):

    img_tone = np.ones((256,256,3), np.uint8)
    img_tone *= 255

    for ch in range(len(tones)):

        color = (0, 0, 0)

        if ch == 0:
            if len(tones) == 3:
                color = (255, 0, 0)
        elif ch == 1:
            color = (0, 255, 0)
        elif ch == 2:
            color = (0, 0, 255)

        for i in range(1, 256):
            x0 = i - 1
            y0 = 255 - tones[ch][i-1]
            x1 = i
            y1 = 255 - tones[ch][i]
            cv2.line(img_tone, (x0, y0), (x1, y1), color, 1)

    cv2.imshow('tone curve', img_tone)

argv = sys.argv
argc = len(argv)

print('%s applys tone cuve to the image' % argv[0])
print('[usage] python %s <image> <tone curve equation for all channels>' % argv[0])
print('[usage] python %s <image> <tone curve equation for blue> <... for green> <... for red>' % argv[0])

print('the equation is expressed using I:  -1 <= I <= 1')

if argc < 3:
    print('no tone curves specified')
    quit()

if argc != 3 and argc != 5:
    print('specify 1 tone curve or 3 tone curves')
    quit()

src = cv2.imread(argv[1])

if src is None:
    print('failed to load image: %s' % argv[1])
    quit()

cv2.imshow('original', src)

I = np.linspace(-1,1,256)

tones = []

for i in range(2,argc):
    tone = eval(argv[i])
    tone_min = np.min(tone)
    tone_max = np.max(tone)

    """
    if tone_min < 0:
        tone -= tone_min

    if tone_max > tone_min:
        tone /= (tone_max - tone_min) 

    """

    tone = np.clip(tone * 255, 0, 255).astype(np.uint8)

    tones.append(tone)

display_tones(tones)

clone = src.copy()

b, g, r = cv2.split(clone)

if len(tones) == 1:
    dst_b  = cv2.LUT(b, tones[0]) 
    dst_g  = cv2.LUT(g, tones[0]) 
    dst_r  = cv2.LUT(r, tones[0]) 
else:
    dst_b  = cv2.LUT(b, tones[0]) 
    dst_g  = cv2.LUT(g, tones[1]) 
    dst_r  = cv2.LUT(r, tones[2]) 

dst = cv2.merge([dst_b, dst_g, dst_r])
cv2.imshow('tone curved', dst)

print('Hit s-key to save and terminate.')
print('Hit any other key to quit.')

key = cv2.waitKey(0)
cv2.destroyAllWindows()

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_tone_curved.png' % filename
    cv2.imwrite(dst_path, dst)
    print('save %s' % dst_path)