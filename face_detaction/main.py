import cv2
import numpy as np
import torch
import serial
import time
import threading
from ais_bench.infer.interface import InferSession
from my_deep_sort import DeepSort
from det_utils import letterbox, scale_coords, nms, scale_coords_landmarks

# 导入新模块
import shared_state as state
import web_server

# --- 局部全局变量 (仅 AI 线程内部使用) ---
last_locked_bbox = None 

# --- 辅助函数 ---

def preprocess_image(image, cfg, bgr2rgb=True):
    """图片预处理"""
    img, scale_ratio, pad_size = letterbox(image, new_shape=cfg['input_shape'])
    if bgr2rgb:
        img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    return img, scale_ratio, pad_size

def calculate_iou(box1, box2):
    """计算 IoU"""
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])

    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0

def draw_tracked_bbox(img, bbox, identities=None, offset=(0, 0), ser=None):
    # 引用 shared_state 中的变量
    global last_locked_bbox
    
    current_frame_detections = [] 
    
    # 获取交互状态
    current_roi = None
    current_locked_id = None
    with state.interaction_lock:
        current_roi = state.user_roi
        current_locked_id = state.locked_track_id

        # 检查 ROI 是否超时
        if current_locked_id is None and current_roi is not None:
            if state.roi_timeout_counter > 0:
                state.roi_timeout_counter -= 1
            else:
                state.user_roi = None
                current_roi = None
                print("Search timeout. ROI cleared.")

    if current_locked_id is None and current_roi is None:
        with state.detection_lock:
            state.latest_detections = []
        last_locked_bbox = None
        return img

    found_target_id = current_locked_id
    target_bbox_current_frame = None

    # --- ID 丢失找回 (IoU) ---
    if current_locked_id is not None and identities is not None:
        if current_locked_id not in identities and last_locked_bbox is not None:
            best_iou = 0
            best_match_id = None
            for i, box in enumerate(bbox):
                x1, y1, x2, y2 = [int(b) for b in box]
                current_box = [x1, y1, x2, y2]
                iou = calculate_iou(last_locked_bbox, current_box)
                if iou > 0.3 and iou > best_iou:
                    best_iou = iou
                    best_match_id = int(identities[i])
            
            if best_match_id is not None:
                print(f"⚠️ ID Switch: {current_locked_id} -> {best_match_id}")
                found_target_id = best_match_id
                with state.interaction_lock:
                    state.locked_track_id = best_match_id
                current_locked_id = best_match_id 

    # --- 遍历绘制 ---
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        id = int(identities[i]) if identities is not None else 0
        
        is_target = False

        if current_locked_id is not None:
            if id == current_locked_id:
                is_target = True
        elif current_roi is not None:
            ux1, uy1, ux2, uy2 = current_roi
            if ux1 < center_x < ux2 and uy1 < center_y < uy2:
                found_target_id = id
                is_target = True

        if is_target:
            target_bbox_current_frame = [x1, y1, x2, y2]
            current_frame_detections.append({
                'bbox': (x1, y1, x2, y2),
                'id': id,
                'center': (center_x, center_y)
            })
            if ser is not None:
                try:
                    data_str = f"{id},{center_x},{center_y}\n"
                    ser.write(data_str.encode('utf-8'))
                except Exception as e:
                    print(f"Serial error: {e}")

    if target_bbox_current_frame is not None:
        last_locked_bbox = target_bbox_current_frame
    
    if current_locked_id is None and found_target_id is not None:
        with state.interaction_lock:
            state.locked_track_id = found_target_id
            state.user_roi = None 

    with state.detection_lock:
        state.latest_detections = current_frame_detections
        
    return img

def infer_frame_with_vis(image, model, labels_dict, cfg, deepsort, ser=None, bgr2rgb=True):
    """单帧推理与可视化"""
    img, scale_ratio, pad_size = preprocess_image(image, cfg, bgr2rgb)
    if len(img.shape) == 3: img = img[None]
    output = model.infer([img])[0]
    output = torch.tensor(output)
    boxout = nms(output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"], nm=10)
    pred_all = boxout[0].numpy()
    scale_coords(cfg['input_shape'], pred_all, image.shape, ratio_pad=(scale_ratio, pad_size))

    xywh_bboxs = []
    confs = []
    clses = []
    
    if pred_all.shape[0] > 0:
        for i in range(pred_all.shape[0]):
            x1, y1, x2, y2 = pred_all[i, :4]
            conf = pred_all[i, 4]
            cls = pred_all[i, 5]
            xywh_bboxs.append([x1, y1, x2-x1, y2-y1])
            confs.append(conf)
            clses.append(cls)
            
        xywh_bboxs = torch.Tensor(xywh_bboxs)
        confs = torch.Tensor(confs)
        clses = torch.Tensor(clses)
        
        outputs = deepsort.update(xywh_bboxs, confs, clses, image)
        if isinstance(outputs, list): outputs = np.array(outputs)
            
        if outputs is not None and len(outputs) > 0 and not isinstance(outputs, tuple):
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            img_vis = draw_tracked_bbox(image, bbox_xyxy, identities, ser=ser)
        else:
            img_vis = image
            with state.detection_lock:
                state.latest_detections = []
    else:
        with state.detection_lock:
            state.latest_detections = []
        img_vis = image
    return img_vis

def ai_inference_loop(model_path, deepsort_path, labels_dict, cfg, ser):
    print("AI Inference thread started.")
    try:
        model = InferSession(0, model_path)
        deepsort = DeepSort(model_path=deepsort_path, max_dist=0.4, min_confidence=0.3, 
                            nms_max_overlap=0.5, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Model loading failed: {e}")
        return

    while True:
        img_for_ai = None
        with state.buffer_lock:
            if state.frame_buffer is not None:
                img_for_ai = state.frame_buffer.copy()
        
        if img_for_ai is None:
            time.sleep(0.1)
            continue

        try:
            infer_frame_with_vis(img_for_ai, model, labels_dict, cfg, deepsort, ser=ser)
        except Exception as e:
            print(f"AI Error: {e}")

def main():
    # 1. 配置参数
    cfg = { 'conf_thres': 0.4, 'iou_thres': 0.5, 'input_shape': [640, 640] }
    model_path = 'model/yolo_face.om'
    deepsort_path = 'model/deepsort_mars.om'
    SERIAL_PORT = '/dev/ttyAMA0' 
    labels_dict = {0: 'face'}

    # 2. 初始化串口
    ser = None
    # try: ser = serial.Serial(SERIAL_PORT, 115200, timeout=0.1)
    # except: print("Serial init failed")

    # 3. 启动 Flask 线程 (调用 web_server 模块)
    flask_thread = threading.Thread(target=web_server.run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    # 4. 启动 AI 线程
    ai_thread = threading.Thread(target=ai_inference_loop, args=(model_path, deepsort_path, labels_dict, cfg, ser))
    ai_thread.daemon = True
    ai_thread.start()

    # 5. 主线程：摄像头采集
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    print("System Ready. Press Ctrl+C to stop.")
    try:
        while True:
            is_cam_on = True
            with state.state_lock:
                is_cam_on = state.camera_enabled

            if not is_cam_on:
                time.sleep(0.5)
                continue

            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue
            
            with state.buffer_lock:
                state.frame_buffer = frame
            
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        if ser: ser.close()

if __name__ == "__main__":
    main()