import cv2 as cv
import numpy as np
import torch
from torchvision import transforms as T

def draw_on_src(img_src, boxes, labels, scores=None):
    # 左上角x 左上角y 框宽 框高 confidence
    for box, label in zip(boxes, labels):
        print(box)
        # 根据输入来进行调整绘制参数
        # cv.rectangle(img_src, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 2)
        cv.rectangle(img_src, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)


def to_even(number):
    if number & 1:
        return number + 1
    return number

def format_img(np_img):
    # YOLO 输入的预处理，最好直接在GPU上进行处理
    im = torch.from_numpy(np_img).to(torch.device('cuda:0')).permute(2, 0, 1).float() / 255.0
    _, _H, _W = im.shape
    # BGR2RGB
    im = im[[2, 1, 0], ...]
    # 1.最大等比例缩放某一个边到640  2.填充到640x640
    # 1.计算缩放比例
    factor = max(_W / 640, _H / 640)  # default 640
    target_W, target_H = to_even(int(_W / factor)), to_even(int(_H / factor))
    # 2.计算填充值
    dif_w = int((640 - target_W) / 2)
    dif_h = int((640 - target_H) / 2)
    # 定义处理算子
    ts = T.Compose([
        T.Resize((target_H, target_W)),
        T.Pad([dif_w, dif_h], fill=0.7)
    ])
    im = ts(im)
    # 返回一些预处理后的图像, 和预处理时附加的信息
    return im, dif_w, dif_h, factor

def gstreamer_pipeline(
        capture_width=1280,  # 摄像头预捕获的图像宽度
        capture_height=720,  # 摄像头预捕获的图像高度
        display_width=1280,  # 窗口显示的图像宽度
        display_height=720,  # 窗口显示的图像高度
        framerate=30,  # 捕获帧率
        flip_method=4,  # 是否旋转图像
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            ))
