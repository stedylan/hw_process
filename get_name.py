import torch
from hwdb import HWDB
from model import ConvNet
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt


def predict(img):
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        # transforms.RandomCrop(64, padding=8),
        transforms.Grayscale()
    ])

    # img = cv2.imread('pic/da.png')
    plt.imshow(img)
    # plt.imshow(img)
    img_ = transform2(img)
    img_ = img_.unsqueeze(0).cuda()
    # img = cv2.resize(img, (64, 64))
    # img_ = torch.from_numpy(img).float().cuda().unsqueeze(0).unsqueeze(0)/255
    # img

    with torch.no_grad():
        y = net(img_)
        idx = y.argmax(1)
        c_b = d[idx]
        b = bytes.fromhex(c_b)
        char = b.decode('gb2312')
        print(char)

# load model


d = []
index = 0
g = os.walk('data/train')
for path, dir_list, file_list in g:
    dir_list.sort()
    for dir_name in dir_list:
        d.append(dir_name)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomCrop(64, padding=8),
    transforms.ToTensor(),
    transforms.Grayscale()
])

batch_size = 1
lr = 0.01

dataset = HWDB(path='data', transform=transform)
num_classes = dataset.num_classes
trainloader, testloader = dataset.get_loader(batch_size)

net = ConvNet(num_classes).to('cuda')
net.load_state_dict(torch.load('checkpoints/handwriting_iter_019.pth'))
net.eval()
# print("net loaded...")

# 找出框

# for i in range(10):
#     for j in range(10):
# Load the image
img = cv2.imread('pic/test.png')

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(img, 200, 200)
# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(
    edged, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 1)
thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# apply some dilation and erosion to join the gaps - change iteration to detect more or less area's

kernel = np.ones((9, 9), np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations=3)  # 膨胀，kernel所有值取最大
thresh = cv2.erode(thresh, kernel, iterations=3)  # 腐蚀，kernel所有制取最小

# Find the contours
contours, hierarchy = cv2.findContours(thresh,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

# 边太多的形状不考虑，只考虑四边形，对应coutours的长度为8，实际情况有干扰的话，可能不止8
rec_cnt = []
for i in range(len(contours)):
    if len(contours[i]) >= 8 and len(contours[i]) <= 10:
        # cv2.drawContours(img, contours, i, (0, 255, 0), 2)
        rec_cnt.append(contours[i])
# 原始绘制形状
for cnt in rec_cnt:
    x, y, w, h = cv2.boundingRect(cnt)
    if w*h > 500:
        # predict(img[x:x+w, y:y+h])
        cv2.imwrite('test.png', img)
        # cv2.rectangle(img,
        #               (x, y), (x+w, y+h),
        #               (0, 255, 0),
        #               2)
