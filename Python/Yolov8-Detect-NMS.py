import tensorrt as trt
trt.init_libnvinfer_plugins(None, "")  # 自定义的NMS要加上这句话
import time
import torch
import cv2 as cv
from collections import namedtuple, OrderedDict
import numpy as np
from utils import format_img, draw_on_src


class LoadDetectNMS:
    def __init__(self, model_path, confidence=0.6, nms_thresh=0.4):
        # 加载模型
        self.confidence = confidence
        self.nms_threshold = nms_thresh
        self.input_shape = (1, 3, 640, 640)
        self.N, self.C, self.H, self.W = self.input_shape
        logger = trt.Logger(trt.Logger.INFO)
        device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.bindings = OrderedDict()
        self.bindings_addrs = OrderedDict()
        self.context = None
        with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = model.get_binding_shape(index)
                data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            self.bindings_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
            self.context = model.create_execution_context()

    def _model_process(self, img_src):
        # 获取处理的结果
        img_blob, dif_w, dif_h, factor = format_img(img_src)
        # print('预处理完毕')
        self.bindings_addrs['images'] = img_blob.data_ptr()
        self.context.execute_v2(list(self.bindings_addrs.values()))
        scores = self.bindings['scores'].data.squeeze()
        flag = scores > self.confidence
        bboxes = self.bindings['bboxes'].data.squeeze()[flag]
        labels = self.bindings['labels'].data.squeeze()[flag]
        scores = scores[flag]
        # 得到x1, y1
        bboxes[:, 0] -= dif_w
        bboxes[:, 1] -= dif_h
        bboxes[:, 2] -= dif_w
        bboxes[:, 3] -= dif_h
        bboxes *= factor
        return bboxes, scores, labels

    def __call__(self, img_src):
        boxes, scores, labels = self._model_process(img_src)
        boxes, labels = boxes.to(torch.int32), labels.to(torch.int32)
        return boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()

def main():
    infer = LoadDetectNMS('yolov8s-detect-nms.engine')
    img = cv.imread("people.png")
    boxes, scores, labels = infer(img)
    draw_on_src(img, boxes, labels, scores=scores)
    cv.imshow("1", img)
    cv.waitKey(0)

if __name__ == '__main__':
    main()