import os
import cv2
import numpy as np
import struct
from tqdm import tqdm

g = os.walk('data/test_gnt')

for root, _, files in g:
    for file in tqdm(files):
        with open(os.path.join(root, file), 'rb') as fp:
            index = 0
            head = True
            while head:
                if not head: # 判断文件是否读到结尾
                    break # 读到文件结尾立即结束
                head = int.from_bytes(fp.read(4), byteorder='little')
                tag_code = fp.read(2)
                tag = tag_code.hex()
                width = int.from_bytes(fp.read(2), byteorder='little')
                height = int.from_bytes(fp.read(2), byteorder='little')
                bitmap = np.frombuffer(fp.read(width*height), np.uint8)
                img = bitmap.reshape((height, width))
                base = 'data/test/' # 起始地址
                dirs = base + tag
                if not os.path.exists(dirs):
                    os.makedirs(dirs)
                if width>0 and height>0:
                    cv2.imwrite(os.path.join(dirs, str(index)+'.png'), img)

                # 文件编号
                index += 1
                # index = index + 4 + 2 + 4 + width*height
                
            
            