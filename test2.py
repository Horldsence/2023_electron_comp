# 边缘检测（Sobel、Laplace、Canny）
import cv2 as cv
from picam import Imget
getImg = Imget()
 
# Sobel一阶微分算子
def Sobel(img):
    # 1、对X和Y方向求微分
    x = cv.Sobel(img, cv.CV_16S, 1,     0)
    y = cv.Sobel(img, cv.CV_16S, 0,     1)
    #                 深度      x方向阶数 y方向阶数

    # 2、取绝对值
    absX = cv.convertScaleAbs(x)  # 转回uint8
    absY = cv.convertScaleAbs(y)

    # 3、线性混合
    dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    #                          比例       比例  常数

    # 4、显示
    cv.imshow("absX", absX)
    cv.imshow("absY", absY)
    cv.imshow("dst", dst)
    return dst

#定义形状检测函数
def ShapeDetection(img, imgContour):
    contours,hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)  #寻找轮廓点
    for obj in contours:
        area = cv.contourArea(obj)  #计算轮廓内区域的面积
        cv.drawContours(imgContour, obj, -1, (255, 0, 0), 4)  #绘制轮廓线
        perimeter = cv.arcLength(obj,True)  #计算轮廓周长
        approx = cv.approxPolyDP(obj,0.02*perimeter,True)  #获取轮廓角点坐标
        CornerNum = len(approx)   #轮廓角点的数量
        x, y, w, h = cv.boundingRect(approx)  #获取坐标值和宽度、高度

        #轮廓对象分类
        if CornerNum ==3: objType ="triangle"
        elif CornerNum == 4:
            if w==h: objType= "Square"
            else:objType="Rectangle"
        elif CornerNum>4: objType= "Circle"
        else:objType="N"

        cv.rectangle(imgContour,(x,y),(x+w,y+h),(0,0,255),2)  #绘制边界框
        cv.putText(imgContour,objType,(x+(w//2),y+(h//2)),cv.FONT_HERSHEY_COMPLEX,0.6,(0,0,0),1)  #绘制文字
    return imgContour

while True:
    # 读取图片
    img = getImg.getImg()
    imgGray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)  #转灰度图
    imgBlur = cv.GaussianBlur(imgGray,(5,5),1)  #高斯模糊
    cv.imshow("img", imgBlur)

    imgSobel = Sobel(imgBlur)         #Sobel一阶微分算子
    imgSobel = cv.GaussianBlur(imgSobel,(5,5),1)  #高斯模糊
    imgCanny = cv.Canny(imgSobel,60,60)  #Canny算子边缘检测
    imgContour = img.copy()
    cv.imshow("canny", imgCanny)
    imgProc = ShapeDetection(imgCanny, imgContour)
    cv.imshow("shape", imgProc)
    cv.waitKey(1)