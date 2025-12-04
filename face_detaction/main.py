# 导入代码依赖
import cv2
import numpy as np
import torch
import serial
import time
from ais_bench.infer.interface import InferSession
import threading
from flask import Flask, Response
from my_deep_sort import DeepSort
from det_utils import letterbox, scale_coords, nms, scale_coords_landmarks


# --- 辅助函数 ---

def preprocess_image(image, cfg, bgr2rgb=True):
    """图片预处理"""
    img, scale_ratio, pad_size = letterbox(image, new_shape=cfg['input_shape'])
    if bgr2rgb:
        img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)  # HWC2CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    return img, scale_ratio, pad_size

def draw_tracked_bbox(img, bbox, identities=None, offset=(0, 0), ser=None):
    """绘制追踪框并通过串口发送数据"""
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        
        # 计算中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        id = int(identities[i]) if identities is not None else 0
        
        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制 ID 和 中心点
        label = f'ID:{id} ({center_x},{center_y})'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        
        # [新增] 串口发送逻辑
        # TODO
        # if ser is not None and ser.isOpen():
        #     try:
        #         # 协议格式: "ID,X,Y\n" (例如: "1,320,240\n")
        data_str = f"{id},{center_x},{center_y}\n"
        #         ser.write(data_str.encode('utf-8'))
        print(f"Sent: {data_str.strip()}") # 调试用
        #     except Exception as e:
        #         print(f"Serial send error: {e}")
        
    return img

def infer_frame_with_vis(image, model, labels_dict, cfg, deepsort, ser=None, bgr2rgb=True):
    """单帧推理与可视化"""
    img, scale_ratio, pad_size = preprocess_image(image, cfg, bgr2rgb)
    
    if len(img.shape) == 3:
        img = img[None]

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
            
            w = x2 - x1
            h = y2 - y1
            xywh_bboxs.append([x1, y1, w, h])
            confs.append(conf)
            clses.append(cls)
            
        xywh_bboxs = torch.Tensor(xywh_bboxs)
        confs = torch.Tensor(confs)
        clses = torch.Tensor(clses)
        
        # DeepSort 更新
        outputs = deepsort.update(xywh_bboxs, confs, clses, image)
        
        if isinstance(outputs, list):
            outputs = np.array(outputs)
            
        if outputs is not None and len(outputs) > 0 and not isinstance(outputs, tuple):
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            # 传入串口对象 ser
            img_vis = draw_tracked_bbox(image, bbox_xyxy, identities, ser=ser)
        else:
            img_vis = image
    else:
        img_vis = image

    return img_vis




frame_buffer = None  # 用于存储最新的原始帧
buffer_lock = threading.Lock() # 线程锁，防止读写冲突
app = Flask(__name__) # Flask 应用

def generate_raw_stream():
    global frame_buffer, buffer_lock
    while True:
        with buffer_lock:
            if frame_buffer is None:
                continue
            # 拷贝一份，避免编码时被修改
            current_frame = frame_buffer.copy()
        
        # 编码为 JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", current_frame)
        if not flag:
            continue
        
        # 输出字节流
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')
        
        # 控制推流帧率，避免占用过多 CPU (例如限制在 30fps)
        time.sleep(0.03)

@app.route("/")
def video_feed():
    return Response(generate_raw_stream(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")


def ai_inference_loop(model_path, deepsort_path, labels_dict, cfg, ser):
    global frame_buffer, buffer_lock
    print("AI Inference thread started.")
    
    print("Loading models in AI thread...")
    try:
        
        model = InferSession(0, model_path)
        
        deepsort = DeepSort(model_path=deepsort_path, max_dist=0.4, min_confidence=0.3, 
                            nms_max_overlap=0.5, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100)
        print("Models loaded successfully in AI thread.")
    except Exception as e:
        print(f"CRITICAL ERROR: Model loading failed: {e}")
        return

    while True:
        # 1. 获取最新帧
        img_for_ai = None
        with buffer_lock:
            if frame_buffer is not None:
                img_for_ai = frame_buffer.copy() # 拷贝一份给 AI 用
        
        if img_for_ai is None:
            time.sleep(0.1)
            continue

        # 2. 执行推理
        try:
            infer_frame_with_vis(img_for_ai, model, labels_dict, cfg, deepsort, ser=ser)
        except Exception as e:
            print(f"AI Error: {e}")

# --- 主程序 ---
def main():
    global frame_buffer
    
    # 1. 配置参数
    cfg = {
        'conf_thres': 0.4,
        'iou_thres': 0.5,
        'input_shape': [640, 640],
    }
    model_path = 'model/yolo_face.om'
    deepsort_path = 'model/deepsort_mars.om'
    SERIAL_PORT = '/dev/ttyAMA0' 
    labels_dict = {0: 'face'}


    # 2. 初始化串口
    ser = None
    # try:
    #     ser = serial.Serial(SERIAL_PORT, 115200, timeout=0.1)
    # except:
    #     print("Serial init failed")

    # 3. 启动 Flask 线程 (守护线程)
    flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False))
    flask_thread.daemon = True
    flask_thread.start()

    # 4. 启动 AI 线程 (守护线程)
    ai_thread = threading.Thread(target=ai_inference_loop, args=(model_path, deepsort_path, labels_dict, cfg, ser))
    ai_thread.daemon = True
    ai_thread.start()

    # 5. 主线程：专门负责读取摄像头 (采集线程)
    camera_source = 0 
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    print("Camera capture started. Press Ctrl+C to stop.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed.")
                time.sleep(1)
                continue
            
            # 更新全局帧
            with buffer_lock:
                frame_buffer = frame
            
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        if ser: ser.close()

if __name__ == "__main__":
    main()