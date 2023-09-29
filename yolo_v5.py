import numpy as np
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes, xyxy2xywh

dict_object = {
    0: 'bicycle',
    1: 'car',
    2: 'motorcycle',
    3: 'bus',
    4: 'truck',
    5: 'lorry',
    6: 'trailer'
}


class YOLOv5:

    def __init__(self, path: str, img_size=None, device=torch.device("cuda:0"), classes=None):
        self.device = device
        # load FP32 model
        self.model = attempt_load(path, device=self.device, fuse=False)
        # model stride
        self.stride = int(self.model.stride.max())
        if img_size is None:
            img_size = [640, 640]
        self.img_size = img_size
        self.classes = classes
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *img_size).to(self.device).type_as(next(self.model.parameters())))  # run once

    @torch.no_grad()
    def predict(
            self,
            im0s,
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
    ):
        # Convert
        img = letterbox(im0s, self.img_size, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        pred = self.model(img, augment=augment)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, self.classes, agnostic_nms, max_det=max_det)

        classes = []
        deth = pred[0]
        bbox = []
        conf = []

        if deth is not None and len(deth):
            deth[:, :4] = scale_boxes(img.shape[2:], deth[:, :4], im0s.shape).round()
            for *xyxy, conf1, cls in reversed(deth):
                if int(cls) in dict_object.keys():
                    xywh = xyxy
                    xywh[-2] = xyxy[-2] - xyxy[0]
                    xywh[-1] = xyxy[-1] - xyxy[1]
                    conf.append(float(conf1.cpu()))
                    classes.append(int(cls.cpu()))
                    bbox.append([int(i.cpu()) for i in xywh])

        return bbox, conf, classes
