 
import os
import sys
import numpy as np
import cv2
 
IMAGE_SIZE = 64
 
#返回目前库中的人数
def get_length(path_name) :
    images,labels = read_path(path_name)    
    #将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
    #IMAGE_SIZE为64，故尺寸为1503 * 64 * 64 * 3
    #图片为64 * 64像素,一个像素3个颜色值(RGB)
    images = np.array(images)
    
    #标注数据，'zx'文件夹下都是我的脸部图像，全部指定为0，另外一个文件夹下是同学的，全部指定为1
    label_last = []
    label_last2 = []
    for each in labels :
        each_last = each.split('\\')[-1]
        label_last.append(each_last)
        for each1 in label_last :
            if each1 not in label_last2:
                label_last2.append(each_last)
    length = len(label_last2)

    return length

#返回目前库中的人名列表
def get_names(path_name) :
    images,labels = read_path(path_name)    
    #将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
    #IMAGE_SIZE为64，故尺寸为1503 * 64 * 64 * 3
    #图片为64 * 64像素,一个像素3个颜色值(RGB)
    images = np.array(images)
    
    #标注数据，'zx'文件夹下都是我的脸部图像，全部指定为0，另外一个文件夹下是同学的，全部指定为1
    label_last = []
    label_last2 = []
    for each in labels :
        each_last = each.split('\\')[-1]
        label_last.append(each_last)
        for each1 in label_last :
            if each1 not in label_last2:
                label_last2.append(each_last)

    return label_last2

#按照指定图像大小调整尺寸
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #获取图像尺寸
    h, w, _ = image.shape
    
    #对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)    
    
    #计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    
    #RGB颜色
    BLACK = [0, 0, 0]
    
    #给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    #调整图像大小并返回
    return cv2.resize(constant, (height, width))
 
#读取训练数据
images = []
labels = []
def read_path(path_name):    
    for dir_item in os.listdir(path_name):
        #从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        
        if os.path.isdir(full_path):    #如果是文件夹，继续递归调用
            read_path(full_path)
        else:   #文件
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)                
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                
                #放开这个代码，可以看到resize_image()函数的实际调用效果
                #cv2.imwrite('1.jpg', image)
                
                images.append(image)                
                labels.append(path_name)                                
                    
    return images,labels
    

#从指定路径读取训练数据
def load_dataset(path_name):
    images,labels = read_path(path_name)    
    #将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
    #IMAGE_SIZE为64，故尺寸为1503 * 64 * 64 * 3
    #图片为64 * 64像素,一个像素3个颜色值(RGB)
    images = np.array(images)
    print(images.shape)    
    
    #标注数据，'zx'文件夹下都是我的脸部图像，全部指定为0，另外一个文件夹下是同学的，全部指定为1
    label_last = []
    label_last2 = []
    for each in labels :
        each_last = each.split('\\')[-1]
        label_last.append(each_last)
        for each1 in label_last :
            if each1 not in label_last2:
                label_last2.append(each_last)
    length = len(label_last2)
    print(label_last2)
    labels = np.array([label_last2.index(label) if label_last2.index(label)+1 else length for label in label_last])   
    print(labels)
    return images, labels
 
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s path_name\r\n" % (sys.argv[0])) 
    else:
        images, labels = load_dataset("C:\\Users\\HP\\Desktop\\faceweb\\data")
