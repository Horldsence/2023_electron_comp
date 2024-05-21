import torch
from procFunc.mathAssoc import mathProc
from basicFunc.yoloLibrary import DetectMultiBackend,LoadImages,Annotator,non_max_suppression,scale_coords,colors
mtProc = mathProc()
# yolo模型加载
device = torch.device('cpu')
model = DetectMultiBackend("yoloModel/best-small.pt", device=device, dnn=False)
if model.pt:
    model.model.float()
print("模型加载完成")
def detect_img(img0): #预测
    device = torch.device('cpu')
    stride, names = model.stride, model.names  #读取模型的步长和模型的识别类别
    centerPointList = []
    dataset = LoadImages(img0=img0, img_size=[640, 640], stride=stride, auto=False)  #对读取的图片进行格式化处理
    # print(dataset)
    for im, im0s in dataset:
        im = (torch.from_numpy(im).to(device).float()/255)[None]  #把图片数据转换成张量
        pred = model(im, augment=False, visualize=False)   #进行检测
        det = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)[0]  #对检测结果进行处理

        im0 = im0s.copy()
        annotator = Annotator(im0, line_width=0, example=str(names))

        data = dict.fromkeys(names, 0)
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()  #识别结果都在这里了

            for *xyxy, conf, cls in reversed(det):   #xyxy是识别结果的标注框的坐标，conf是识别结果的置信度，cls是识别结果对应的类别
                xyxy_int = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                cicrleCenter = mtProc.calcDotCenter(xyxy_int)
                centerPointList.append((int(cls), cicrleCenter))
                c = int(cls)
                if names[c] in data:
                    data[names[c]] += 1
                else:
                    data[names[c]] = 1
                label = (f'{names[c][0]}{data[names[c]]}')
                annotator.box_label(xyxy, label, color=colors(c, True))  #对图片进行标注，就是画框

        im0 = annotator.result()
        return data, im0, centerPointList
