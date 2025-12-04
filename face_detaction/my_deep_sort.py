import numpy as np
from ais_bench.infer.interface import InferSession
import torch
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import cv2

# --- DeepSort 类 ---
class DeepSort:
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, 
                 nms_max_overlap=1.0, max_iou_distance=0.7, 
                 max_age=70, n_init=3, nn_budget=100, use_cuda=False):
        
        self.session = InferSession(0, model_path)
        
        # 1. 自动解析模型输入
        try:
            input_desc = self.session.get_inputs()[0]
            self.model_shape = tuple(input_desc.shape)
            
            shape_size = np.prod(self.model_shape)
            if input_desc.size == shape_size:
                self.model_dtype = np.uint8
            else:
                self.model_dtype = np.float32
                
        except:
            self.model_shape = (16, 128, 64, 3)
            self.model_dtype = np.float32

        self.batch_size = self.model_shape[0]
        
        # 2. 解析 Layout (NHWC vs NCHW)
        if len(self.model_shape) == 4 and self.model_shape[3] == 3:
            self.layout = 'NHWC'
            self.input_h = self.model_shape[1]
            self.input_w = self.model_shape[2]
        elif len(self.model_shape) == 4 and self.model_shape[1] == 3:
            self.layout = 'NCHW'
            self.input_h = self.model_shape[2]
            self.input_w = self.model_shape[3]
        else:
            self.layout = 'NCHW'
            self.input_h = 128
            self.input_w = 64
            
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        self.min_confidence = min_confidence

    def _preprocess_batch(self, im_crops):
        batch_data = np.zeros(self.model_shape, dtype=self.model_dtype)
        
        for i, crop in enumerate(im_crops):
            if i >= self.batch_size: break
            
            # Resize
            crop = cv2.resize(crop, (self.input_w, self.input_h))
            
            # BGR -> RGB
            crop = crop[:, :, ::-1]
            
            # Layout 转换
            if self.layout == 'NCHW':
                crop = crop.transpose(2, 0, 1)
            
            if self.model_dtype == np.uint8:
                # 模型需要 uint8，保持 0-255
                crop = crop.astype(np.uint8)
            else:
                # 模型需要 float32，归一化到 0.0-1.0
                crop = crop.astype(np.float32) / 255.0
            
            batch_data[i] = crop
            
        return np.ascontiguousarray(batch_data)

    def update(self, bbox_xywh, confidences, classes, ori_img):
        self.height, self.width = ori_img.shape[:2]
        
        features = []
        if isinstance(confidences, torch.Tensor): confidences = confidences.cpu().numpy()
        if isinstance(bbox_xywh, torch.Tensor): bbox_xywh = bbox_xywh.cpu().numpy()
            
        indices = [i for i, c in enumerate(confidences) if c > self.min_confidence]
        bbox_xywh = bbox_xywh[indices]
        confidences = confidences[indices]
        
        if len(bbox_xywh) > 0:
            crops = []
            for box in bbox_xywh:
                x, y, w, h = box
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(self.width, int(x+w)), min(self.height, int(y+h))
                crop = ori_img[y1:y2, x1:x2]
                if crop.size == 0: continue
                crops.append(crop)
            
            if crops:
                num_crops = len(crops)
                num_batches = int(np.ceil(num_crops / self.batch_size))
                batch_features = []
                
                for b in range(num_batches):
                    start = b * self.batch_size
                    end = min((b + 1) * self.batch_size, num_crops)
                    input_data = self._preprocess_batch(crops[start:end])
                    output = self.session.infer([input_data.copy()])[0]
                    batch_features.append(output[:end-start])
                
                if batch_features:
                    features = np.vstack(batch_features)
                else:
                    features = np.array([])
            else:
                features = np.array([])
        
        detections = []
        for i, (box, conf) in enumerate(zip(bbox_xywh, confidences)):
            if i < len(features):
                detections.append(Detection(box, conf, features[i].flatten()))

        self.tracker.predict()
        self.tracker.update(detections)

        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1: continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3])
            outputs.append([x1, y1, x2, y2, track.track_id])
            
        return np.array(outputs)