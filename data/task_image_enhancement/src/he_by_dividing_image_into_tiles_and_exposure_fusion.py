import sys, cv2
import numpy as np

NR_DIVS = 3
CLIP=2.0

argv = sys.argv
argc = len(argv)

if argc < 2:
    print('%s divides the image into tiles and equalizes histogram and fuses' % argv[0])
    print('%s <image> [<number of divisions(4)>]' % argv[0])
    quit()

src = cv2.imread(argv[1])
height, width, nrChrs = src.shape[:3]

nrDivs = NR_DIVS
if argc > 2:
    nrDivs = int(argv[2])

TILE_SIZE_HORZ = width // nrDivs
TILE_SIZE_VERT = height // nrDivs

nrTilesHorz = width // TILE_SIZE_HORZ

if width % TILE_SIZE_HORZ:
    nrTilesHorz += 1

nrTilesVert = height // TILE_SIZE_VERT

if height % TILE_SIZE_VERT:
    nrTilesVert += 1

images = []

no = 1

for Y in range(nrTilesVert):
    print('processing block %d / %d' % (Y+1, nrTilesVert))

    for X in range(nrTilesHorz):

        bgr = cv2.split(src)
        for ch in range(nrChrs):

            histogram = np.zeros((256), np.int32)
            newValue = np.zeros((256), np.int32)
            nrPixels = 0

            for y in range(TILE_SIZE_VERT):

                yy = Y * TILE_SIZE_VERT + y
                if yy >= height - 1:
                    break

                for x in range(TILE_SIZE_HORZ):

                    xx = X * TILE_SIZE_HORZ + x
                    if xx >= width - 1:
                        continue

                    idx = bgr[ch][yy][xx]
                    histogram[idx] += 1
                    nrPixels += 1

            th = CLIP * nrPixels // 100
            sum = 0
            clipped = 0

            for idx in range(256):
                if histogram[idx] > th:
                    clipped += histogram[idx] - th
                    histogram[idx] = th
        
            for idx in range(256):
                sum += histogram[idx] + clipped // 256
                if nrPixels > 0:
                    newValue[idx] = 255 * sum // nrPixels

            for y in range(height):
                for x in range(width):
                    idx = bgr[ch][y][x]
                    bgr[ch][y][x] = newValue[idx]
    
        dst = cv2.merge(bgr)

        dst_path = '%04d.png' % no
        no += 1
        cv2.imwrite(dst_path, dst)
        print('save %s' % dst_path)        

        images.append(dst)

# Merge using Exposure Fusion
print("Merging using Exposure Fusion ... ");
mergeMertens = cv2.createMergeMertens()
exposureFusion = mergeMertens.process(images)

cv2.imshow('fusion', exposureFusion)
cv2.waitKey(0)

# Save output image
print("Saving output ... fusion.png")
cv2.imwrite("fusion.png", exposureFusion * 255)

