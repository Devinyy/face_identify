#-*- coding: utf-8 -*-
 
import cv2
import sys
import gc
from face_train import Model
from load_data import get_names, get_length
 
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
    
    label_last2 = get_names("C:\\Users\\HP\\Desktop\\faceweb\\data")
    length = get_length("C:\\Users\\HP\\Desktop\\faceweb\\data")
    #加载模型
    model = Model()
    model.load_model(file_path = './model/zx.face.model.h5')    
              
    #框住人脸的矩形边框颜色       
    color = (0, 255, 0)
    
    #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)
    
    #人脸识别分类器本地存储路径
    cascade_path = "S:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml"    
    #定义分类器（人眼识别）
    eye_cascade = cv2.CascadeClassifier("S:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml")
    #定义分类器（嘴巴识别）
    mouth_cascade =cv2.CascadeClassifier("S:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_mcs_mouth.xml")
    
    #循环检测识别人脸
    while True:
        ret, frame = cap.read()   #读取一帧视频
        
        if ret is True:
            #图像灰化，降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)                

        #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                
                #截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                result, result2 = model.face_predict(image) 
                print('result:', result)

                count = 0
                for each in result :
                    for each1 in each :
                        if each1 <= 0.72 :
                            count += 1
                        else :
                            pass 
            
                if count == length :
                    result2 = -1 

                faceID = result2         

                print(result2)

                #如果是0号
                if faceID >= 0 :
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    #利用分类器识别出哪个区域为眼睛
                    eyes = eye_cascade.detectMultiScale(image)  
                    # 框出眼睛
                    for(ex, ey, ew, eh) in eyes:
                        cv2.rectangle(frame, (x - 10 + ex, y - 10 + ey), ( x - 10 + ex + ew, y - 10 + ey + eh), (255, 0 , 0), 1)

                    #利用分类器识别出哪个区域为嘴巴
                    mouth_zone = mouth_cascade.detectMultiScale(image,1.3,3,minSize=(10,10))
                    #框出嘴巴
                    for (ax, ay, aw, ah) in mouth_zone:
                        if ay <= (h + 10)/2 :
                            pass
                        else :
                            cv2.rectangle(frame, (x - 10 + ax, y - 10 + ay), ( x - 10 + ax + aw,  y - 10 + ay + ah), (255, 0, 0), 1)

                    #文字提示是谁
                    cv2.putText(frame,label_last2[faceID], 
                                (x + 35, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽                                                    
                else :
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    #文字提示是谁
                    cv2.putText(frame,'???', 
                                (x + 35, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽

        cv2.imshow("recognize me!", frame)
        
        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break
 
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
