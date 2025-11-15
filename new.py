import pylab
from PIL import Image
from matplotlib import pyplot as plt

img = Image.open('/home/hmx/hmx1123/datasets/interhand2.6m/test/hms/1_0_left.jpg')  # 读取图像
plt.imshow(img)
pylab.show()  # 加上这句才会显示，对应需要import pylab

imagePixels = img.size # 获取图像大小
w = imagePixels[0]  # python下标从0开始
print(w)  # 输出图像的宽度，即列数
print(imagePixels)  # 输出图像大小
