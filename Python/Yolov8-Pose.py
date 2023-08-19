from collections import namedtuple, OrderedDict
import time
import cv2 as cv
import numpy as np
import tensorrt as trt
import torch
from torchvision import ops
from utils import format_img

KPS_COLORS = [[0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0],
              [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0],
              [255, 128, 0], [255, 128, 0], [51, 153, 255], [51, 153, 255],
              [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255]]

SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]]

LIMB_COLORS = [[51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],
               [255, 51, 255], [255, 51, 255], [255, 51, 255], [255, 128, 0],
               [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0],
               [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0],
               [0, 255, 0], [0, 255, 0]]


class LoadPoseEngine:
    def __init__(self, model_path, confidence=0.6, nms_thresh=0.4):
        device = torch.device('cuda:0')  # default
        logger = trt.Logger(trt.Logger.INFO)
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        # 加载模型
        self.confidence = confidence
        self.nms_threshold = nms_thresh
        self.input_shape = (1, 3, 640, 640)  # default
        self.N, self.C, self.H, self.W = self.input_shape
        self.bindings = OrderedDict()
        self.bindings_addrs = OrderedDict()
        self.context = None
        self.num_nodes = 17  # 关节点的个数
        self.dif_w, self.dif_h, self.factor = 0, 0, 1
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
        self.dif_w, self.dif_h, self.factor = dif_w, dif_h, factor
        # print('预处理完毕')
        self.bindings_addrs['images'] = img_blob.data_ptr()
        self.context.execute_v2(list(self.bindings_addrs.values()))
        out_prob = self.bindings['output0'].data.squeeze().permute(1, 0)

        # 以下这段在GPU上操作, 并且要用矩阵操作, 可以节约时间, 处理成NMSBOXES函数可接受的数据格式
        flag = out_prob[:, 4] > self.confidence
        out_prob = out_prob[flag]
        # 得到x1, y1
        out_prob[:, 0] -= out_prob[:, 2] / 2 + dif_w
        out_prob[:, 1] -= out_prob[:, 3] / 2 + dif_h
        # 得到x2, y2
        out_prob[:, 2] += out_prob[:, 0]
        out_prob[:, 3] += out_prob[:, 1]
        out_prob[:, :4] *= factor
        # 原3,4为宽高 现得到x2,y2 根据不同的nms算法的输入参数来进行调整
        bboxes = out_prob[:, :4]
        scores = out_prob[:, 4]
        points = out_prob[:, 5:]
        return bboxes, scores, points  # 处理完后把数据做最后的NMS处理

    def _nms(self, bboxes, scores, points):
        indices = ops.nms(bboxes, scores, self.nms_threshold)
        # print('NMS处理完成')
        bboxes = bboxes[indices]
        scores = scores[indices]
        points = points[indices]
        # 处理关节点的正确位置
        for i in range(self.num_nodes):
            points[:, i * 3] = (points[:, i * 3] - self.dif_w) * self.factor
            points[:, i * 3 + 1] = (points[:, i * 3 + 1] - self.dif_h) * self.factor
            points[:, i * 3: i * 3 + 2] = torch.round(points[:, i * 3: i * 3 + 2])
        return torch.round(bboxes), scores, points

    def __call__(self, img_src):
        self._model_process(img_src)
        bboxes, scores, points = self._model_process(img_src)
        bboxes, scores, points = self._nms(bboxes, scores, points)
        return bboxes, scores, points

    def draw_pose(self, np_img, bboxes, scores, points):
        for (bbox, score, kpt) in zip(bboxes, scores, points):
            cv.rectangle(np_img, bbox[:2], bbox[2:], (0, 0, 255), 2)
            for i in range(19):
                if i < 17:
                    px, py, ps = kpt[i * 3: i * 3 + 3]
                    if ps > self.confidence:
                        kcolor = KPS_COLORS[i]
                        px, py = round(px), round(py)
                        cv.circle(np_img, (px, py), 3, kcolor, -1)
                xi, yi = SKELETON[i]
                pos1_s = kpt[(xi - 1) * 3 + 2]
                pos2_s = kpt[(yi - 1) * 3 + 2]
                if pos1_s > self.confidence and pos2_s > self.confidence:
                    limb_color = LIMB_COLORS[i]
                    pos1_x, pos1_y = round(kpt[(xi - 1) * 3]), round(kpt[(xi - 1) * 3 + 1])
                    pos2_x, pos2_y = round(kpt[(yi - 1) * 3]), round(kpt[(yi - 1) * 3 + 1])
                    cv.line(np_img, (pos1_x, pos1_y), (pos2_x, pos2_y),
                            limb_color, 2)


def main():
    infer = LoadPoseEngine('yolov8s-pose.engine')  # 加载模型
    cap = cv.VideoCapture(0)
    while 1:
        f, np_img = cap.read()
        if not f: break
        s = time.time()
        bboxes, scores, points = infer(np_img)
        e = time.time() + 1e-5  # 防止计算的fps分子为0
        scores = scores.cpu().numpy()
        bboxes = bboxes.to(torch.int32).cpu().numpy()
        points = points.cpu().numpy()  # A x 51条数据 17个点 每个点3个信息 分别是该点的x坐标，y坐标，置信度
        infer.draw_pose(np_img, bboxes, scores, points)
        print(f"fps: {format(1 / (e - s))}")
        cv.imshow("1", np_img)
        cv.waitKey(1)

if __name__ == '__main__':
    main()
