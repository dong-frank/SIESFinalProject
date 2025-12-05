from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import cv2
import time
import threading
import shared_state as state  # 导入共享状态

app = Flask(__name__)
CORS(app)

def generate_raw_stream():
    """生成视频流"""
    while True:
        # 1. 检查摄像头是否开启
        is_cam_on = True
        with state.state_lock:
            is_cam_on = state.camera_enabled
            
        if not is_cam_on:
            time.sleep(0.5)
            continue

        # 2. 获取帧
        current_frame = None
        with state.buffer_lock:
            if state.frame_buffer is None:
                time.sleep(0.05)
                continue
            current_frame = state.frame_buffer.copy()
        
        # 3. 编码
        (flag, encodedImage) = cv2.imencode(".jpg", current_frame)
        if not flag:
            continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')
        
        time.sleep(0.03)

# --- 路由定义 ---

@app.route("/")
def index():
    return "<h1>Atlas Video Server is Running</h1><p>Stream at: <a href='/video_feed'>/video_feed</a></p>"

@app.route("/video_feed")
def video_feed():
    return Response(generate_raw_stream(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/toggle_camera", methods=['POST'])
def toggle_camera():
    try:
        data = request.json
        target = bool(data.get('enable'))
        with state.state_lock:
            state.camera_enabled = target
        return jsonify({"status": "success", "camera_enabled": state.camera_enabled})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 400

@app.route("/get_status", methods=['GET'])
def get_status():
    with state.state_lock:
        return jsonify({"camera_enabled": state.camera_enabled})

@app.route("/set_target_region", methods=['POST'])
def set_target_region():
    try:
        data = request.json
        x1 = int(data.get('x1'))
        y1 = int(data.get('y1'))
        x2 = int(data.get('x2'))
        y2 = int(data.get('y2'))
        
        with state.interaction_lock:
            state.user_roi = (x1, y1, x2, y2)
            state.locked_track_id = None
            state.roi_timeout_counter = 2 # 2帧寿命
            
        print(f"User selected region: {state.user_roi}")
        return jsonify({"status": "success", "msg": "Region received"})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 400

@app.route("/cancel_tracker", methods=['POST'])
def cancel_tracker():
    with state.interaction_lock:
        state.user_roi = None
        state.locked_track_id = None
    print("Tracking canceled by user.")
    return jsonify({"status": "success", "msg": "Tracker reset"})

@app.route("/get_detections", methods=['GET'])
def get_detections():
    data = []
    with state.detection_lock:
        data = state.latest_detections
    return jsonify(data)

def run_flask_app(host="0.0.0.0", port=5000):
    """启动 Flask 应用"""
    app.run(host=host, port=port, debug=False, use_reloader=False)