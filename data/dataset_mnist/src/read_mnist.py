import numpy as np
import gzip

def bytes2data(b):

    data = b[3] + b[2] * 256 + b[1] * 256 * 256 + b[0] * 256 * 256 * 256
    return data 

def read_data(data_path):

    with  gzip.open(data_path, mode='rb') as f:
    
        # read signature
        b = f.read(4)
        signature = bytes2data(b)
       
        if signature != 0x803 and signature != 0x801:
            print('failed to read signature: %x' % signature)
            return
    
        else:
            print('succeeded to read signature: %x' % signature)
    
        # read number of data
        b = f.read(4)
        nr_data = bytes2data(b)
        print('number of data:', nr_data)

        if signature == 0x803: # Image
        
            # read height 
            b = f.read(4)
            height = bytes2data(b)
    
            # read width
            b = f.read(4)
            width = bytes2data(b)
    
            print('%d x %d' % (width, height)) 

            images = np.frombuffer(f.read(), np.uint8, offset = 0)
            images = images.reshape((nr_data, height, width)) 
         
            dst_path = '_images.npy'

            if nr_data == 60000:
                dst_path = 'train' + dst_path
            elif nr_data == 10000:
                dst_path = 'test' + dst_path
            else:
                dst_path = 'unknown' + dst_path

            np.save(dst_path, images)
            print('save %s' % dst_path)
            print(images.shape)

        else: # Labels

            labels = np.frombuffer(f.read(), np.uint8, offset = 0)
            
            dst_path = '_labels.npy'

            if nr_data == 60000:
                dst_path = 'train' + dst_path
            elif nr_data == 10000:
                dst_path = 'test' + dst_path
            else:
                dst_path = 'unknown' + dst_path

            np.save(dst_path, labels)
            print('save %s' % dst_path)
            print(labels.shape)

import sys

def main():

    argv = sys.argv
    argc = len(argv)

    print('%s read MNIST dataset' % argv[0])
    print('[usage] python %s <MNIST gz file>' % argv[0])

    if argc < 2:
        quit()
        
    read_data(argv[1])

if __name__ == '__main__':
    main()

