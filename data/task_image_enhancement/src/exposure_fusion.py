import cv2
import numpy as np
import os, sys, glob

def usage(progName):
    print('%s fusions images' % progName)
    print('%s <image1> <image2> ....' % progName)
    print('%s <wildcard for images: ex) *.png>' % progName)

def main():

    argv = sys.argv
    argc = len(argv)
    usage(argv[0])

    needsAlignment = False
    images = []

    if argc > 2:
        for filename in argv[1:]:
            im = cv2.imread(filename)
            images.append(im)
    elif argc == 2:
        filenames = glob.glob(argv[1])
        print('fusing..')
        for filename in filenames:
            print(filename)
            im = cv2.imread(filename)
            images.append(im)
    else:
        quit()

    # Align input images
    if needsAlignment:
        print("Aligning images ... ")
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(images, images)
    else :
        print("Skipping alignment ... ")
  
    # Merge using Exposure Fusion
    print("Merging using Exposure Fusion ... ");
    mergeMertens = cv2.createMergeMertens()
    exposureFusion = mergeMertens.process(images)

    cv2.imshow('fusion', exposureFusion)
    print('Hit any key to terminate')
    cv2.waitKey(0)

    # Save output image
    exposureFusion = np.clip(exposureFusion * 255.0, 0, 255)
    exposureFusion = exposureFusion.astype(np.uint8)

    print("Saving resulting fusion image ... fusion.png")
    cv2.imwrite("fusion.png", exposureFusion)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
