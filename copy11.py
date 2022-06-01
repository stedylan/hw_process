import os
import shutil
from tqdm import tqdm

# cnt = 0
# for root, dirs, files in tqdm(os.walk('data/train')):
#     for name in files:
#         fn = str(cnt)+'.png'
#         shutil.copyfile(os.path.join(root, name), os.path.join('/home/yuhang/repo/PaddleOCR/train_data/rec/train/', fn))
#         cnt += 1

# cnt = 0
# for root, dirs, files in os.walk('data/test'):
#     for name in files:
#         fn = str(cnt)+'.png'
#         shutil.copyfile(os.path.join(root, name), os.path.join('/home/yuhang/repo/PaddleOCR/train_data/rec/test/', fn))
#         cnt += 1
def main():
    fr = open('rec_gt_train.txt', 'w+', encoding='utf-8')
    cnt = 0
    for root, dirs, files in tqdm(os.walk('data/train')):
        for name in files:
            fn = str(cnt)+'.png'
            fr.write(os.path.join('rec/train/', fn))
            fr.write('\t')
            b = root.split('/')[2]
            # print(b)
            if b[0]<='7':
                fr.write(bytes.fromhex(b[0:2]).decode('gbk'))
            else:
                fr.write(bytes.fromhex(b).decode('gbk'))
            fr.write('\n')
            cnt += 1

main()